#include "CLRL/clRL.hpp"
#include <iostream>
#include <numeric>

#define BUFFER(elements) cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * elements)
#define LEAKY_RELU_KERNEL kernels[0]
#define EXPLORATION_KERNEL kernels[1]
#define LOSS_KERNEL kernels[2]

namespace CLRL
{
  // Important variables for the library
  cl::Context context;
  cl::CommandQueue queue;
  cl::vector<cl::Kernel> kernels;
  std::mt19937 gen;
  std::vector<float> ones;
  std::vector<float> a_batched;
  std::vector<size_t> bias_offsets;
  std::vector<size_t> outputs_offsets;

  // Global Getters
  cl::vector<cl::Kernel> getKernels() { return kernels; }
  cl::Context getContext() { return context; }
  cl::CommandQueue getQueue() { return queue; }
  std::mt19937 getGen() { return gen; }
  std::vector<float> getOnes() { return ones; }
  std::vector<float> getABatched() { return a_batched; }
  std::vector<size_t> getBiasOffsets() { return bias_offsets; }
  std::vector<size_t> getOutputsOffsets() { return outputs_offsets; }

  // Other functions
  void createKernels(const cl::Device &device)
  {
#include "CLRL/clRL.opencl"
    cl::Program program = cl::Program(context, source);

    program.build("-cl-std=CL3.0");

    program.createKernels(&kernels);
  }

  // Global Setters
  void setKernels(const cl::vector<cl::Kernel> &val) { kernels = val; }
  void setContext(const cl::Context &val) { context = val; }
  void setQueue(const cl::CommandQueue &val) { queue = val; }
  void setGen(const std::mt19937 &val) { gen = val; }
  void setOnes(const std::vector<float> &val) { ones = val; }
  void setABatched(const std::vector<float> &val) { a_batched = val; }
  void setBiasOffsets(const std::vector<size_t> &val) { bias_offsets = val; }
  void setOutputsOffsets(const std::vector<size_t> &val) { outputs_offsets = val; }

  // Layer constructors
  Layer::Layer(const uint &neuron_num, const uint &input_num, const Activation &activation, const size_t &batch_size) : neurons(neuron_num), input_num(input_num), activation(activation), biases(BUFFER(neurons)), bias_derivatives(BUFFER(neurons)), weights(BUFFER(neurons * input_num)), weight_derivatives(BUFFER(neurons * input_num)), outputs(BUFFER(batch_size * neurons)), costs(BUFFER(batch_size * neurons)) // Sets the variables for the layer
  {
    // Uses He Initialization for the layer
    float *new_weights = new float[input_num * neurons];
    std::normal_distribution<float> distribution(0.0f, 2.0f / input_num);

    for (size_t i = 0; i < input_num * neurons; i++)
    {
      new_weights[i] = distribution(gen);
    }

    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * input_num * neurons, new_weights);

    // Memory cleanup
    delete[] new_weights;

    // Sets other variables to 0
    queue.enqueueFillBuffer(biases, 0.0f, 0, sizeof(float) * neurons);
    queue.enqueueFillBuffer(bias_derivatives, 0.0f, 0, sizeof(float) * neurons);
    queue.enqueueFillBuffer(weight_derivatives, 0.0f, 0, sizeof(float) * neurons * input_num);
  }

  Layer::Layer(std::ifstream &file, const size_t &batch_size)
  {
    // Read format for the file: neurons, input_num, activation, biases, weights

    // Read vital stuff that can be directly read
    file.read(reinterpret_cast<char *>(&neurons), sizeof(uint));
    file.read(reinterpret_cast<char *>(&input_num), sizeof(uint));
    file.read(reinterpret_cast<char *>(&activation), sizeof(int));

    // Use information and create buffers
    biases = BUFFER(neurons);
    bias_derivatives = BUFFER(neurons);
    weights = BUFFER(input_num * neurons);
    weight_derivatives = BUFFER(input_num * neurons);
    outputs = BUFFER(batch_size * neurons);
    costs = BUFFER(batch_size * neurons);

    // Gather the information that must be copied (weights and biases)
    float *read_data = new float[input_num * neurons];

    file.read(reinterpret_cast<char *>(read_data), sizeof(float) * neurons);
    queue.enqueueWriteBuffer(biases, CL_TRUE, 0, sizeof(float) * neurons, read_data);

    file.read(reinterpret_cast<char *>(read_data), sizeof(float) * input_num * neurons);
    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * input_num * neurons, read_data);

    // Fill the derivative buffers with 0s
    queue.enqueueFillBuffer(bias_derivatives, 0.0f, 0, sizeof(float) * neurons);
    queue.enqueueFillBuffer(weight_derivatives, 0.0f, 0, sizeof(float) * input_num * neurons);

    // Memory cleanup
    delete[] read_data;
  }

  // Layer functionality
  cl::Buffer Layer::forwardPropagation(const cl::Buffer &inputs, const size_t &batch_size)
  {
    cl_command_queue temp_queue = queue();

    clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                  neurons, batch_size, input_num,
                  1.0f,
                  weights(), 0, input_num,
                  inputs(), 0, batch_size,
                  0.0f, outputs(), 0, batch_size,
                  &temp_queue);

    clblast::AxpyBatched(neurons, ones.data(), biases(), bias_offsets.data(), 1, outputs(), outputs_offsets.data(), batch_size, batch_size, &temp_queue);

    switch (activation)
    {
    case LINEAR:
    {
      queue.enqueueFillBuffer(costs, 1.0f, 0, sizeof(float) * neurons * batch_size);
      break;
    }
    case LEAKY_RELU:
    {
      LEAKY_RELU_KERNEL.setArg(0, outputs);
      LEAKY_RELU_KERNEL.setArg(1, costs);
      queue.enqueueNDRangeKernel(LEAKY_RELU_KERNEL, 0, neurons * batch_size);
      break;
    }
    }

    return outputs;
  }

  void Layer::backwardPropagation(const cl::Buffer &inputs, const cl::Buffer &previous_layer_costs,
                                  const float &a, const float &b, const size_t &batch_size)
  {
    cl_command_queue temp_queue = queue();

    // Calculate weight derivatives
    clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
                  neurons, input_num, batch_size,
                  a_batched[0],
                  costs(), 0, batch_size,
                  inputs(), 0, batch_size,
                  b, weight_derivatives(), 0, input_num,
                  &temp_queue);

    // Calculate bias derivatives
    clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
    clblast::AxpyBatched(neurons, a_batched.data(), costs(), outputs_offsets.data(), batch_size, bias_derivatives(), bias_offsets.data(), 1, batch_size, &temp_queue);

    // Calculate previous layer costs
    clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
                  input_num, batch_size, neurons,
                  1.0f,
                  weights(), 0, input_num,
                  costs(), 0, batch_size,
                  0.0f, previous_layer_costs(), 0, batch_size,
                  &temp_queue);

    // Update weights
    clblast::Axpy(neurons * input_num, -a, weight_derivatives(), 0, 1, weights(), 0, 1, &temp_queue);

    // Update biases
    clblast::Axpy(neurons, -a, bias_derivatives(), 0, 1, biases(), 0, 1, &temp_queue);
  }

  void Layer::backwardPropagation(const cl::Buffer &inputs, const float &a, const float &b, const size_t &batch_size)
  {
    cl_command_queue temp_queue = queue();

    // Calculate weight derivatives
    clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
                  neurons, input_num, batch_size,
                  a_batched[0],
                  costs(), 0, batch_size,
                  inputs(), 0, batch_size,
                  b, weight_derivatives(), 0, input_num,
                  &temp_queue);

    // Calculate bias derivatives
    clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
    clblast::AxpyBatched(neurons, a_batched.data(), costs(), outputs_offsets.data(), batch_size, bias_derivatives(), bias_offsets.data(), 1, batch_size, &temp_queue);

    // Update weights
    clblast::Axpy(neurons * input_num, -a, weight_derivatives(), 0, 1, weights(), 0, 1, &temp_queue);

    // Update biases
    clblast::Axpy(neurons, -a, bias_derivatives(), 0, 1, biases(), 0, 1, &temp_queue);
  }

  void Layer::save(std::ofstream &file)
  {
    // Write format for the file: neurons, input_num, activation, biases, weights

    // Write vital stuff that can be directly write
    file.write(reinterpret_cast<char *>(&neurons), sizeof(uint));
    file.write(reinterpret_cast<char *>(&input_num), sizeof(uint));
    file.write(reinterpret_cast<char *>(&activation), sizeof(int));

    // Gather the information that must be written (weights and biases) and write it
    float *write_data = new float[input_num * neurons];

    queue.enqueueReadBuffer(biases, CL_TRUE, 0, sizeof(float) * neurons, write_data);
    file.write(reinterpret_cast<char *>(write_data), sizeof(float) * neurons);

    queue.enqueueReadBuffer(weights, CL_TRUE, 0, sizeof(float) * input_num * neurons, write_data);
    file.write(reinterpret_cast<char *>(write_data), sizeof(float) * input_num * neurons);

    // Memoru cleanup
    delete[] write_data;
  }

  void Layer::useDifferentBatchSize(const size_t &batch_size)
  {
    outputs = BUFFER(batch_size * neurons);
    costs = BUFFER(batch_size * neurons);
  }

  // Layer Getters
  uint Layer::getNeurons() const { return neurons; }
  uint Layer::getInputNum() const { return input_num; }
  Activation Layer::getActivation() const { return activation; }
  cl::Buffer Layer::getBiases() const { return biases; }
  cl::Buffer Layer::getBiasDerivatives() const { return bias_derivatives; }
  cl::Buffer Layer::getWeights() const { return weights; }
  cl::Buffer Layer::getWeightDerivatives() const { return weight_derivatives; }
  cl::Buffer Layer::getOutputs() const { return outputs; }
  cl::Buffer Layer::getCosts() const { return costs; }

  // Layer Setters
  void Layer::setNeurons(const uint &val) { neurons = val; }
  void Layer::setInputNum(const uint &val) { input_num = val; }
  void Layer::setActivation(const Activation &val) { activation = val; }
  void Layer::setBiases(const cl::Buffer &val) { biases = val; }
  void Layer::setBiasDerivatives(const cl::Buffer &val) { bias_derivatives = val; }
  void Layer::setWeights(const cl::Buffer &val) { weights = val; }
  void Layer::setWeightDerivatives(const cl::Buffer &val) { weight_derivatives = val; }
  void Layer::setOutputs(const cl::Buffer &val) { outputs = val; }
  void Layer::setCosts(const cl::Buffer &val) { costs = val; }

  // Agent constructors
  Agent::Agent(const std::vector<uint> &architecture, const std::vector<Activation> &activations, const uint &initial_input_num, const size_t &batch_size) : layers(architecture.size())
  {
    // Layers initialization
    layers[0] = Layer(architecture[0], initial_input_num, activations[0], batch_size);
    for (size_t i = 1; i < architecture.size(); i++)
    {
      layers[i] = Layer(architecture[i], architecture[i - 1], activations[i], batch_size);
    }

    // Global important variable initialization
    ones = std::vector<float>(batch_size, 1.0f);
    bias_offsets = std::vector<size_t>(batch_size, 0);
    outputs_offsets = std::vector<size_t>(batch_size);
    std::iota(outputs_offsets.begin(), outputs_offsets.end(), 0);
    gen = std::mt19937(1);
  }

  Agent::Agent(const std::string &file_name, const size_t &batch_size)
  {
    // Layers initialization
    std::ifstream file(file_name, std::ios::binary);
    size_t layer_num;

    file.read(reinterpret_cast<char *>(&layer_num), sizeof(layer_num));

    layers = std::vector<Layer>(layer_num);
    for (size_t i = 0; i < layer_num; i++)
    {
      layers[i] = Layer(file, batch_size);
    }

    // Global important variable initialization
    ones = std::vector<float>(batch_size, 1.0f);
    bias_offsets = std::vector<size_t>(batch_size, 0);
    outputs_offsets = std::vector<size_t>(batch_size);
    std::iota(outputs_offsets.begin(), outputs_offsets.end(), 0);
    gen = std::mt19937(1);
  }

  // Agent functionality
  void Agent::train(const size_t &epochs, const size_t &batch_size, clEnvironment::Environment &env, const float &a, const float &b)
  {
    // Vars
    cl::Buffer outputs;
    cl::Buffer otherOutputs;
    cl::Buffer actions = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * batch_size);
    cl::Buffer otherActions = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * batch_size);
    cl::Buffer previous_states;
    cl::Event event;
    cl_command_queue temp_queue = queue();

    const uint num_outputs = layers[layers.size() - 1].getNeurons();

    std::uniform_int_distribution<int> seeds(0, 200);
    a_batched = std::vector<float>(batch_size, (1.0f - b));

    float averaged_a = a / batch_size;

    // Execution
    for (size_t i = 0; i < epochs; i++)
    {
      previous_states = cl::Buffer(env.getStates());
      if (i && layers.size() == 1)
        event.wait();

      // Get actions
      outputs = layers[0].forwardPropagation(previous_states, batch_size);
      if (i)
        event.wait();
      for (size_t j = 1; j < layers.size(); j++)
      {
        outputs = layers[j].forwardPropagation(outputs, batch_size);
      }

      for (size_t j = 0; j < batch_size; j++)
      {
        clblast::Max<float>(num_outputs, actions(), j, outputs(), j, batch_size, &temp_queue);
      }

      // Exploration vs exploitation trade off
      EXPLORATION_KERNEL.setArg(0, actions);
      EXPLORATION_KERNEL.setArg(1, seeds(gen));
      EXPLORATION_KERNEL.setArg(2, num_outputs);
      queue.enqueueNDRangeKernel(EXPLORATION_KERNEL, 0, batch_size);

      // Update environment
      env.updateStates(actions);

      // Run again but no trade off and no update
      otherOutputs = layers[0].forwardPropagation(env.getStates(), batch_size);
      for (size_t j = 1; j < layers.size(); j++)
      {
        otherOutputs = layers[j].forwardPropagation(otherOutputs, batch_size);
      }

      for (size_t j = 0; j < batch_size; j++)
      {
        clblast::Max<float>(num_outputs, otherActions(), j, otherOutputs(), j, batch_size, &temp_queue);
      }

      queue.enqueueFillBuffer(layers[layers.size() - 1].getCosts(), 0.0f, 0, sizeof(float) * num_outputs * batch_size);

      // Get costs for output layer
      LOSS_KERNEL.setArg(0, otherOutputs);
      LOSS_KERNEL.setArg(1, outputs);
      LOSS_KERNEL.setArg(2, env.getRewards());
      LOSS_KERNEL.setArg(3, layers[layers.size() - 1].getCosts());
      LOSS_KERNEL.setArg(4, otherActions);
      LOSS_KERNEL.setArg(5, actions);
      LOSS_KERNEL.setArg(6, batch_size);
      queue.enqueueNDRangeKernel(LOSS_KERNEL, 0, batch_size, cl::NullRange, nullptr, &event);

      // Back prop
      for (size_t j = layers.size() - 1; j > 0; j--)
      {
        layers[j].backwardPropagation(layers[j - 1].getOutputs(), layers[j - 1].getCosts(), averaged_a, b, batch_size);
      }
      layers[0].backwardPropagation(previous_states, averaged_a, b, batch_size);
    }

    queue.finish();
  }

  std::string Agent::test(const size_t &epochs, const size_t &batch_size, clEnvironment::Environment &env)
  {
    // Vars
    cl::Buffer outputs;
    cl::Buffer actions = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint) * batch_size);

    const uint num_outputs = layers[layers.size() - 1].getNeurons();

    cl_command_queue temp_queue = queue();

    float *outs = new float[num_outputs];

    // Execution
    for (size_t i = 0; i < epochs; i++)
    {
      outputs = layers[0].forwardPropagation(env.getStates(), batch_size);
      for (size_t j = 1; j < layers.size(); j++)
      {
        outputs = layers[j].forwardPropagation(outputs, batch_size);
      }

      for (size_t j = 0; j < batch_size; j++)
      {
        clblast::Max<float>(num_outputs, actions(), j, outputs(), j, batch_size, &temp_queue);
      }

      env.updateStates(actions);
    }

    delete[] outs;

    // Get data
    float *final_rewards = new float[batch_size];

    queue.enqueueReadBuffer(env.getRewards(), CL_TRUE, 0, sizeof(float) * batch_size, final_rewards);

    std::string output = "";

    for (size_t i = 0; i < batch_size; i++)
    {
      output += "Reward for agent " + std::to_string(i) + " is " + std::to_string(final_rewards[i]) + "\n";
    }

    delete[] final_rewards;

    return output;
  }

  void Agent::save(const std::string &file_name)
  {
    std::ofstream file(file_name, std::ios::binary);
    size_t layer_num = layers.size();

    file.write(reinterpret_cast<char *>(&layer_num), sizeof(layer_num));

    for (size_t i = 0; i < layer_num; i++)
    {
      layers[i].save(file);
    }
  }

  void Agent::changeBatchSize(const size_t &batch_size)
  {
    // Global important variable initialization
    ones = std::vector<float>(batch_size, 1.0f);
    bias_offsets = std::vector<size_t>(batch_size, 0);
    outputs_offsets = std::vector<size_t>(batch_size);
    std::iota(outputs_offsets.begin(), outputs_offsets.end(), 0);

    // Update layers
    for (auto layer : layers)
    {
      layer.useDifferentBatchSize(batch_size);
    }
  }

  // Agent getters
  std::vector<Layer> Agent::getLayers() const { return layers; }

  // Agent setters
  void Agent::setLayers(const std::vector<Layer> &val) { layers = val; }
}