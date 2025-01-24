std::string source = R"(
	__kernel void ouptutAddBias(__global float *outputs, __global const float *biases, const size_t batch_size, const size_t neurons) 
	{
		if (get_global_id(0) < batch_size && get_global_id(1) < neurons)
			outputs[get_global_linear_id(0)] += biases[get_global_id(1)];
	}

	__kernel void subtract(__global float *params, __global float *derivs, const size_t num_params)
	{
		const size_t gid = get_global_id(0);
		if (gid < num_params)
			params[gid] += derivs[gid];
	}
)";