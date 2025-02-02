#ifndef CLRL_HPP
#define CLRL_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <clblast.h>
#include <clEnvironment.h>
#include <random>
#include <fstream>

#define get(type, var) type get##var() const
#define set(type, var) void set##var(const type &val)

namespace CLRL
{
    typedef enum Activations
    {
        LINEAR,
        LEAKY_RELU
    } Activation;

    typedef unsigned int uint;

    // Gets important variables for the library
    cl::vector<cl::Kernel> getKernels();
    cl::Context getContext();
    cl::CommandQueue getQueue();
    std::mt19937 getGen();
    std::vector<float> getOnes();
    std::vector<float> getABatched();
    std::vector<size_t> getBiasOffsets();
    std::vector<size_t> getOutputsOffsets();

    // Other functions
    void createKernels(const cl::Device &device); // Create the kernels that are used for the library (Context must be set first)

    // Sets important variables for the library
    set(cl::vector<cl::Kernel>, Kernels); // Allows manual setting of the kernels used for the library
    set(cl::Context, Context);            // Set the context used for the library
    set(cl::CommandQueue, Queue);         // Set the command queue used for the library
    set(std::mt19937, Gen);               // Set the random generator used for the library (auto sets with 1 as seed if not manually set)
    set(std::vector<float>, Ones);
    set(std::vector<float>, ABatched);
    set(std::vector<size_t>, BiasOffsets);
    set(std::vector<size_t>, OutputsOffsets);

    class Layer
    {
    private:
        uint neurons;          // Number of neurons in the layer
        uint input_num;        // Number of inputs for the layer
        Activation activation; // Activation of the layer

        cl::Buffer biases;             // Vector of dimensions neurons
        cl::Buffer bias_derivatives;   // Vector of dimensions neurons
        cl::Buffer weights;            // Matrix of dimensions neurons x input_num
        cl::Buffer weight_derivatives; // Matrix of dimensions neurons x input_num
        cl::Buffer outputs;            // Matrix of dimensions neurons x batch_size
        cl::Buffer costs;              // Matrix of dimensions neurons x batch_size

    public:
        // Basic constructors
        Layer() = default;
        Layer(const uint &neuron_num, const uint &input_num, const Activation &activation, const size_t &batch_size); // Regular constructor
        Layer(std::ifstream &file, const size_t &batch_size);                                                         // Loads the layer from a binary file

        // Functions for functionality
        cl::Buffer forwardPropagation(const cl::Buffer &inputs, const size_t &batch_size); // Forward propagation function for the layer
        void backwardPropagation(const cl::Buffer &inputs, const cl::Buffer &previous_layer_costs,
                                 const float &a, const float &b, const size_t &batch_size);                           // Peforms backward propagation for any layer other than the FIRST layer of the neural network. Excluding the input layer.
        void backwardPropagation(const cl::Buffer &inputs, const float &a, const float &b, const size_t &batch_size); // Performs backward propagtion for the first hidden layer of the neural network.
        void save(std::ofstream &file);                                                                               // Stores the layer to a binary file
        void useDifferentBatchSize(const size_t &batch_size);                                                         // For in case the batch size for the network is changed

        // Getters for every private variable
        get(uint, Neurons);
        get(uint, InputNum);
        get(Activation, Activation);
        get(cl::Buffer, Biases);
        get(cl::Buffer, BiasDerivatives);
        get(cl::Buffer, Weights);
        get(cl::Buffer, WeightDerivatives);
        get(cl::Buffer, Outputs);
        get(cl::Buffer, Costs);

        // Setters for every private variable
        set(uint, Neurons);
        set(uint, InputNum);
        set(Activation, Activation);
        set(cl::Buffer, Biases);
        set(cl::Buffer, BiasDerivatives);
        set(cl::Buffer, Weights);
        set(cl::Buffer, WeightDerivatives);
        set(cl::Buffer, Outputs);
        set(cl::Buffer, Costs);
    };

    class Agent
    {
    private:
        std::vector<Layer> layers;

    public:
        // Basic constructors
        Agent() = default;
        Agent(const std::vector<uint> &architecture, const std::vector<Activation> &activations, const uint &initial_input_num, const size_t &batch_size); // Regular constructor
        Agent(const std::string &file_name, const size_t &batch_size);                                                                                     // Loads the model from a binary file

        // Functions for functionality
        void train(const size_t &epochs, const size_t &batch_size, clEnvironment::Environment &env, const float &a, const float &b);
        void test(const size_t &epochs, const size_t &batch_size, clEnvironment::Environment &env);
        void save(const std::string &file_name);
        void changeBatchSize(const size_t &batch_size);

        // Getters for every private variable
        get(std::vector<Layer>, Layers);

        // Setters for every private variable
        set(std::vector<Layer>, Layers);
    };
}

#undef get
#undef set

#endif