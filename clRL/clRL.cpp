// clRL.cpp : Defines the entry point for the application.
//

#include "clRL.h"
#include <random>

#define FLAGS CL_MEM_READ_WRITE
#define DELTA_ADD_BIAS_KERNEL 0
#define SUBTRACT_KERNEL 1

namespace clRL
{
	cl::Context context;
	cl::CommandQueue queue;
	std::vector<cl::Kernel> kernels;

	Layer::Layer(const size_t& neuron_num, const size_t& input_num, const size_t &batch_size, const unsigned int &seed)
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

		float* weight_values = new float[neurons * inputs];
		std::mt19937 gen(seed);
		std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / inputs));
		for (size_t i = 0; i < neurons * inputs; i++)
		{
			weight_values[i] = dist(gen);
		}
		queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * neurons * inputs, weight_values);
		delete[] weight_values;
	}

	const cl::Buffer& Layer::runLayer(const cl::Buffer& ins, const size_t &batch_size)
	{
		cl_command_queue temp_queue = queue();

		// Calculate dot product between inputs and weights
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
			batch_size, neurons, inputs, 1.0f,
			ins(), 0, inputs,
			weights(), 0, neurons,
			0.0f, outputs(), 0, batch_size,
			&temp_queue
		);

		// Add bias terms
		kernels[DELTA_ADD_BIAS_KERNEL].setArg(0, outputs);
		kernels[DELTA_ADD_BIAS_KERNEL].setArg(1, biases);
		kernels[DELTA_ADD_BIAS_KERNEL].setArg(2, batch_size);
		kernels[DELTA_ADD_BIAS_KERNEL].setArg(3, neurons);
		queue.enqueueNDRangeKernel(kernels[DELTA_ADD_BIAS_KERNEL], 0, cl::NDRange(batch_size, neurons));

		return outputs;
	}
	
	const void Layer::backProp(const cl::Buffer& ins, cl::Buffer& prev_costs, const size_t& batch_size, const float& a, const float& b)
	{
		cl_command_queue temp_queue = queue();

		// Calculate new weight derivatives
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
			inputs, neurons, batch_size, a,
			ins(), 0, batch_size,
			costs(), 0, neurons,
			b, weight_derivatives(), 0, inputs,
			&temp_queue);

		// Calculate previous layer costs
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
			batch_size, inputs, neurons, 1.0f,
			costs(), 0, neurons,
			weights(), 0, neurons,
			0.0f, prev_costs(), 0, batch_size,
			&temp_queue);

		// Calculate new bias derivatives
		clblast::Scal(neurons * batch_size, a, costs(), 0, 1, &temp_queue);
		clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
		for (size_t i = 0; i < neurons; i++)
		{
			clblast::Sum<float>(batch_size, bias_derivatives(), i, costs(), i, neurons, &temp_queue);
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

	const void Layer::backProp(const cl::Buffer& ins, const size_t& batch_size, const float& a, const float& b)
	{
		cl_command_queue temp_queue = queue();

		// Calculate new weight derivatives
		clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
			inputs, neurons, batch_size, a,
			ins(), 0, batch_size,
			costs(), 0, neurons,
			b, weight_derivatives(), 0, inputs,
			&temp_queue);

		// Calculate new bias derivatives
		clblast::Scal(neurons * batch_size, a, costs(), 0, 1, &temp_queue);
		clblast::Scal(neurons, b, bias_derivatives(), 0, 1, &temp_queue);
		for (size_t i = 0; i < neurons; i++)
		{
			clblast::Sum<float>(batch_size, bias_derivatives(), i, costs(), i, neurons, &temp_queue);
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
}