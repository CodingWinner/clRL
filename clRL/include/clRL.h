// clRL.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>
#include <CL/opencl.hpp>
#include <clblast.h>
#include <clEnvironment.h>

namespace clRL
{
	extern cl::Context context;
	extern cl::CommandQueue queue;
	extern std::vector<cl::Kernel> kernels;

	void createKernels();

	class Layer
	{
	private:
		cl::Buffer biases;
		cl::Buffer weights;
		cl::Buffer bias_derivatives;
		cl::Buffer weight_derivatives;
		cl::Buffer outputs;
		cl::Buffer costs;
		size_t neurons;
		size_t inputs;

	public:
		Layer() = default;
		Layer(const size_t& neuron_num, const size_t& input_num, const size_t& batch_size, const unsigned int& seed = 32);

		const cl::Buffer &runLayer(const cl::Buffer& ins, const size_t& batch_size);
		const void backProp(const cl::Buffer& ins, cl::Buffer& prev_costs, const size_t& batch_size, const float& a, const float& b);
		const void backProp(const cl::Buffer& ins, const size_t& batch_size, const float& a, const float& b);

		~Layer() = default;
	};
}