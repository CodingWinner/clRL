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
		size_t inputs;

	public:
		size_t neurons;
		cl::Buffer outputs;
		cl::Buffer costs;

		Layer() = default;
		Layer(const size_t &neuron_num, const size_t &input_num, const size_t &batch_size);
		Layer(const Layer &l);

		const cl::Buffer &runLayer(const cl::Buffer &ins, const size_t &batch_size);
		void backProp(const cl::Buffer &ins, cl::Buffer &prev_costs, const size_t &batch_size, const float &a, const float &b);
		void backProp(const cl::Buffer &ins, const size_t &batch_size, const float &a, const float &b);

		~Layer() = default;
	};

	class Model
	{
	private:
		std::vector<Layer> layers;

	public:
		Model() = default;
		Model(const std::vector<size_t> &neurons, const size_t &initial_input_num, const size_t &batch_size, const unsigned int &seed);
		Model(const Model &m);

		void getCosts(clEnvironment::Environment &env, const size_t &batch_size);
		void train(clEnvironment::Environment &env, const size_t &num_epochs, const size_t &batch_size, const float &a, const float &b);
		void test(clEnvironment::Environment &env, const size_t &num_epochs, const size_t &batch_size, const std::string &file_name);

		~Model() = default;
	};
}