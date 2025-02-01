// clRL.cpp : Defines the entry point for the application.
//

#include "include/clRL.h"
#include <fstream>

#define FLAGS CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
#define OUTPUT_ADD_BIAS_KERNEL 0
#define SUBTRACT_KERNEL 1
#define GET_Q_VAL_KERNEL 2
#define CALCULATE_LOSS_KERNEL 3

namespace clRL
{
	cl::Context context;
	cl::CommandQueue queue;
	std::vector<cl::Kernel> kernels;
	std::mt19937 gen;

	void createKernels()
	{
#include "include/clRL.opencl"
#include "clRL.h"
		cl::Program program = cl::Program(context, source);
		try
		{
			program.build("-cl-std=CL3.0 -cl-finite-math-only");
		}
		catch (cl::Error e)
		{
			std::cout << e.what() << "\nError code: " << e.err() << "\n";
			if (e.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				auto build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
				std::cout << "Build log:\n"
						  << build_log[0].second << "\n";
			}
			return;
		}
		program.createKernels(&kernels);
	}

	Layer::Layer(const size_t &neuron_num, const size_t &input_num, const size_t &batch_size)
	{
		neurons = neuron_num;
		inputs = input_num;
		biases = cl::Buffer(context, FLAGS, sizeof(float) * neurons);
		bias_derivatives = cl::Buffer(context, FLAGS, sizeof(float) * neurons);
		weights = cl::Buffer(context, FLAGS, sizeof(float) * neurons * inputs);
		weight_derivatives = cl::Buffer(context, FLAGS, sizeof(float) * neurons * inputs);
		outputs = cl::Buffer(context, FLAGS, sizeof(float) * neurons * batch_size);
		costs = cl::Buffer(context, FLAGS, sizeof(float) * neurons * batch_size);
		queue.enqueueFillBuffer(biases, 0.0f, 0, sizeof(float) * neurons);
		queue.enqueueFillBuffer(bias_derivatives, 0.0f, 0, sizeof(float) * neurons);
		queue.enqueueFillBuffer(weight_derivatives, 0.0f, 0, sizeof(float) * neurons * inputs);

		float *weight_values = new float[neurons * inputs];
		std::normal_distribution<float> dist(0.0f, 2.0f / inputs);
		for (size_t i = 0; i < neurons * inputs; i++)
		{
			weight_values[i] = dist(gen);
		}
		queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * neurons * inputs, weight_values);
		delete[] weight_values;
	}

	Layer::Layer(const Layer &l)
	{
		biases = cl::Buffer(l.biases);
		weights = cl::Buffer(l.weights);
		outputs = cl::Buffer(l.outputs);
		costs = l.costs;
		neurons = l.neurons;
		inputs = l.inputs;
	}

	Layer::Layer(std::ifstream &file, const size_t &batch_size)
	{
		file.read(reinterpret_cast<char *>(&neurons), sizeof(neurons));
		file.read(reinterpret_cast<char *>(&inputs), sizeof(inputs));

		biases = cl::Buffer(context, FLAGS, sizeof(float) * neurons);
		bias_derivatives = cl::Buffer(context, FLAGS, sizeof(float) * neurons);
		weights = cl::Buffer(context, FLAGS, sizeof(float) * neurons * inputs);
		weight_derivatives = cl::Buffer(context, FLAGS, sizeof(float) * neurons * inputs);
		outputs = cl::Buffer(context, FLAGS, sizeof(float) * neurons * batch_size);
		costs = cl::Buffer(context, FLAGS, sizeof(float) * neurons * batch_size);
		queue.enqueueFillBuffer(biases, 0.0f, 0, sizeof(float) * neurons);
		queue.enqueueFillBuffer(bias_derivatives, 0.0f, 0, sizeof(float) * neurons);
		queue.enqueueFillBuffer(weight_derivatives, 0.0f, 0, sizeof(float) * neurons * inputs);

		float *read_data = new float[inputs * neurons];

		file.read(reinterpret_cast<char *>(read_data), sizeof(float) * neurons);
		queue.enqueueWriteBuffer(biases, CL_TRUE, 0, sizeof(float) * neurons, read_data);

		file.read(reinterpret_cast<char *>(read_data), sizeof(float) * neurons * inputs);
		queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * neurons * inputs, read_data);

		delete[] read_data;
	}

	const cl::Buffer &Layer::runLayer(const cl::Buffer &ins, const size_t &batch_size)
	{
		cl_command_queue temp_queue = queue();

		// Calculate dot product between inputs and weights
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
					  batch_size, neurons, inputs, 1.0f,
					  ins(), 0, inputs,
					  weights(), 0, neurons,
					  0.0f, outputs(), 0, neurons,
					  &temp_queue);

		// Add bias terms
		kernels[OUTPUT_ADD_BIAS_KERNEL].setArg(0, outputs);
		kernels[OUTPUT_ADD_BIAS_KERNEL].setArg(1, biases);
		kernels[OUTPUT_ADD_BIAS_KERNEL].setArg(2, batch_size);
		kernels[OUTPUT_ADD_BIAS_KERNEL].setArg(3, neurons);
		queue.enqueueNDRangeKernel(kernels[OUTPUT_ADD_BIAS_KERNEL], 0, cl::NDRange(batch_size, neurons));

		return outputs;
	}

	void Layer::backProp(const cl::Buffer &ins, cl::Buffer &prev_costs, const size_t &batch_size, const float &a, const float &b)
	{
		cl_command_queue temp_queue = queue();

		// Calculate new weight derivatives
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
					  inputs, neurons, batch_size, a / batch_size,
					  ins(), 0, inputs,
					  costs(), 0, neurons,
					  b, weight_derivatives(), 0, neurons,
					  &temp_queue);

		// Calculate previous layer costs
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
					  batch_size, inputs, neurons, 1.0f,
					  costs(), 0, neurons,
					  weights(), 0, inputs,
					  0.0f, prev_costs(), 0, inputs,
					  &temp_queue);

		// Calculate new bias derivatives
		clblast::Scal(neurons * batch_size, a, costs(), 0, 1, &temp_queue);
		clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
		for (size_t i = 0; i < batch_size; i++)
		{
			clblast::Axpy(neurons, 1.0f / batch_size, costs(), i * neurons, 1, bias_derivatives(), 0, 1, &temp_queue);
		}

		// Update weight derivatives
		kernels[SUBTRACT_KERNEL].setArg(0, weights);
		kernels[SUBTRACT_KERNEL].setArg(1, weight_derivatives);
		kernels[SUBTRACT_KERNEL].setArg(2, neurons * inputs);
		queue.enqueueNDRangeKernel(kernels[SUBTRACT_KERNEL], 0, neurons * inputs);

		// Update bias derivatives
		kernels[SUBTRACT_KERNEL].setArg(0, biases);
		kernels[SUBTRACT_KERNEL].setArg(1, bias_derivatives);
		kernels[SUBTRACT_KERNEL].setArg(2, neurons);
		queue.enqueueNDRangeKernel(kernels[SUBTRACT_KERNEL], 0, neurons);
	}

	void Layer::backProp(const cl::Buffer &ins, const size_t &batch_size, const float &a, const float &b)
	{
		cl_command_queue temp_queue = queue();

		// Calculate new weight derivatives
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
					  inputs, neurons, batch_size, a / batch_size,
					  ins(), 0, inputs,
					  costs(), 0, neurons,
					  b, weight_derivatives(), 0, neurons,
					  &temp_queue);

		// Calculate new bias derivatives
		clblast::Scal(neurons * batch_size, a, costs(), 0, 1, &temp_queue);
		clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
		for (size_t i = 0; i < batch_size; i++)
		{
			clblast::Axpy(neurons, 1.0f / batch_size, costs(), i * neurons, 1, bias_derivatives(), 0, 1, &temp_queue);
		}

		// Update weight derivatives
		kernels[SUBTRACT_KERNEL].setArg(0, weights);
		kernels[SUBTRACT_KERNEL].setArg(1, weight_derivatives);
		kernels[SUBTRACT_KERNEL].setArg(2, neurons * inputs);
		queue.enqueueNDRangeKernel(kernels[SUBTRACT_KERNEL], 0, neurons * inputs);

		// Update bias derivatives
		kernels[SUBTRACT_KERNEL].setArg(0, biases);
		kernels[SUBTRACT_KERNEL].setArg(1, bias_derivatives);
		kernels[SUBTRACT_KERNEL].setArg(2, neurons);
		queue.enqueueNDRangeKernel(kernels[SUBTRACT_KERNEL], 0, neurons);
	}

	void Layer::saveLayer(std::ofstream &file)
	{
		file.write(reinterpret_cast<char *>(&neurons), sizeof(neurons));
		file.write(reinterpret_cast<char *>(&inputs), sizeof(inputs));

		float *write_data = new float[inputs * neurons];

		queue.enqueueReadBuffer(biases, CL_TRUE, 0, sizeof(float) * neurons, write_data);
		file.write(reinterpret_cast<char *>(write_data), sizeof(float) * neurons);

		queue.enqueueReadBuffer(weights, CL_TRUE, 0, sizeof(float) * neurons * inputs, write_data);
		file.write(reinterpret_cast<char *>(write_data), sizeof(float) * neurons * inputs);

		delete[] write_data;
	}

	Layer &Layer::operator=(const Layer &l)
	{
		neurons = l.neurons;
		inputs = l.inputs;

		biases = l.biases;
		bias_derivatives = l.bias_derivatives;
		weights = l.weights;
		weight_derivatives = l.weight_derivatives;

		outputs = l.outputs;
		costs = l.costs;

		return *this;
	}

	Model::Model(const std::vector<size_t> &neurons, const size_t &initial_input_num, const size_t &batch_size, const unsigned int &seed)
	{
		gen = std::mt19937(seed);
		layers = std::vector<Layer>(neurons.size());
		layers[0] = Layer(neurons[0], initial_input_num, batch_size);
		for (size_t i = 1; i < neurons.size(); i++)
		{
			layers[i] = Layer(neurons[i], neurons[i - 1], batch_size);
		}
	}

	Model::Model(const Model &m)
	{
		layers = std::vector<Layer>();
		for (size_t i = 0; i < m.layers.size(); i++)
		{
			layers.push_back(Layer(m.layers[i]));
		}
	}

	Model::Model(const std::string &file_name, const size_t &batch_size)
	{
		std::ifstream file(file_name, std::ios::binary);

		size_t num_layers;
		file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

		layers = std::vector<Layer>(num_layers);

		for (size_t i = 0; i < num_layers; i++)
		{
			layers[i] = Layer(file, batch_size);
		}
	}

	void Model::getCosts(clEnvironment::Environment &env, const size_t &batch_size)
	{
		cl::Buffer temp = layers[0].runLayer(env.getStates(), batch_size);
		cl::Buffer actions(context, FLAGS, batch_size * sizeof(size_t));
		size_t num_outputs = layers[layers.size() - 1].neurons;
		cl_command_queue temp_queue = queue();
		float *acts = new float[num_outputs * batch_size];
		std::uniform_int_distribution<size_t> new_action(0, num_outputs);
		std::uniform_int_distribution<int> do_random(0, 100);

		for (size_t j = 1; j < layers.size(); j++)
		{
			temp = layers[j].runLayer(temp, batch_size);
		}

		for (size_t j = 0; j < batch_size; j++)
		{
			clblast::Max<float>(num_outputs, actions(), j, temp(), j * num_outputs, 1, &temp_queue);
		}

		kernels[GET_Q_VAL_KERNEL].setArg(0, temp);
		kernels[GET_Q_VAL_KERNEL].setArg(1, actions);
		kernels[GET_Q_VAL_KERNEL].setArg(2, env.getRewards());
		kernels[GET_Q_VAL_KERNEL].setArg(3, layers[layers.size() - 1].costs);
		kernels[GET_Q_VAL_KERNEL].setArg(4, num_outputs);
		kernels[GET_Q_VAL_KERNEL].setArg(5, batch_size);
		queue.enqueueNDRangeKernel(kernels[GET_Q_VAL_KERNEL], 0, cl::NDRange(batch_size, num_outputs));
		queue.finish();
		delete[] acts;
	}

	void Model::train(clEnvironment::Environment &env, const size_t &num_epochs, const size_t &batch_size, const float &a, const float &b)
	{
		cl::Buffer temp;
		cl::Buffer prev_states;
		cl::Buffer actions(context, FLAGS, batch_size * sizeof(size_t));
		size_t num_outputs = layers[layers.size() - 1].neurons;
		float *acts = new float[num_outputs * batch_size];
		cl_command_queue temp_queue = queue();
		std::uniform_int_distribution<size_t> new_action(0, num_outputs);
		std::uniform_int_distribution<int> do_random(0, 100);
		for (size_t i = 0; i < num_epochs; i++)
		{
			temp = layers[0].runLayer(env.getStates(), batch_size);
			for (size_t j = 1; j < layers.size(); j++)
			{
				temp = layers[j].runLayer(temp, batch_size);
			}
			prev_states = cl::Buffer(env.getStates());

			for (size_t j = 0; j < batch_size; j++)
			{
				clblast::Max<float>(num_outputs, actions(), j, temp(), j * num_outputs, 1, &temp_queue);
			}

			queue.enqueueReadBuffer(actions, CL_TRUE, 0, sizeof(float) * num_outputs * batch_size, acts);
			for (size_t j = 0; j < batch_size; j++)
			{
				if (do_random(gen) > 95)
				{
					acts[j] = new_action(gen);
				}
			}
			queue.enqueueWriteBuffer(actions, CL_TRUE, 0, sizeof(float) * num_outputs * batch_size, acts);

			env.updateStates(actions);

			Model m = Model(*this);
			m.getCosts(env, batch_size);
			kernels[CALCULATE_LOSS_KERNEL].setArg(0, temp);
			kernels[CALCULATE_LOSS_KERNEL].setArg(1, layers[layers.size() - 1].costs);
			kernels[CALCULATE_LOSS_KERNEL].setArg(2, num_outputs * batch_size);
			queue.enqueueNDRangeKernel(kernels[CALCULATE_LOSS_KERNEL], 0, num_outputs * batch_size);

			for (size_t j = layers.size() - 1; j > 0; j--)
			{
				layers[j].backProp(layers[j - 1].outputs, layers[j - 1].costs, batch_size, a, b);
			}
			layers[0].backProp(prev_states, batch_size, a, b);
		}
		delete[] acts;
	}

	void Model::test(clEnvironment::Environment &env, const size_t &num_epochs, const size_t &batch_size, const std::string &file_name)
	{
		cl::Buffer temp;
		cl::Buffer actions(context, FLAGS, batch_size * sizeof(size_t));
		size_t num_outputs = layers[layers.size() - 1].neurons;
		cl_command_queue temp_queue = queue();
		float *rewards = new float[batch_size];
		float *outputs = new float[batch_size * num_outputs];
		std::ofstream file(file_name);
		for (size_t i = 0; i < num_epochs; i++)
		{
			temp = layers[0].runLayer(env.getStates(), batch_size);
			for (size_t j = 1; j < layers.size(); j++)
			{
				temp = layers[j].runLayer(temp, batch_size);
			}

			for (size_t j = 0; j < batch_size; j++)
			{
				clblast::Max<float>(num_outputs, actions(), j, temp(), j * num_outputs, 1, &temp_queue);
			}
			env.updateStates(actions);

			queue.enqueueReadBuffer(env.getRewards(), CL_TRUE, 0, sizeof(float) * batch_size, rewards);
			queue.enqueueReadBuffer(temp, CL_TRUE, 0, sizeof(float) * batch_size * num_outputs, outputs);

			file << "Target values for iteration " << i << ":\n";
			for (size_t j = 0; j < batch_size; j++)
			{
				file << "{" << outputs[j * num_outputs];
				for (size_t k = 1; k < num_outputs; k++)
				{
					file << ", " << outputs[j * num_outputs + k];
				}
				file << "}\n";
			}

			file << "\nRewards for iteration " << i << ":\n";
			for (size_t j = 0; j < batch_size; j++)
			{
				file << rewards[j] << " ";
			}
			file << "\n-------------------------------------\n";
		}

		delete[] rewards;
		delete[] outputs;
	}

	void Model::saveModel(const std::string &file_name)
	{
		std::ofstream file(file_name, std::ios::binary);

		size_t num_layers = layers.size();
		file.write(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

		for (size_t i = 0; i < num_layers; i++)
		{
			layers[i].saveLayer(file);
		}
	}

	Model &Model::operator=(const Model &m)
	{
		layers = std::vector<Layer>(m.layers.size());

		for (size_t i = 0; i < m.layers.size(); i++)
		{
			layers[i] = m.layers[i];
		}

		return *this;
	}
}
