std::string source = R"(
	__kernel void ouptutAddBias(__global float *outputs, __global const float *biases, const size_t batch_size, const size_t neurons) 
	{
		if (get_global_id(0) < batch_size && get_global_id(1) < neurons)
			outputs[get_global_linear_id()] += biases[get_global_id(1)];
	}

	__kernel void subtract(__global float *params, __global float *derivs, const size_t num_params)
	{
		const size_t gid = get_global_id(0);
		if (gid < num_params)
			params[gid] += derivs[gid];
	}

	__kernel void QVal(__global const float *all_actions, __global const size_t *indices, __global const float *rewards, 
						__global float *costs, const size_t num_outputs, const size_t batch_size) 
						{
							const size_t gid0 = get_global_id(0);
							const size_t gid1 = get_global_id(1);
							if (gid0 < batch_size && gid1 < num_outputs)
							{
								costs[gid0 * num_outputs + gid1] = 0.0f;
								const size_t ridx = indices[gid0];
								if (gid1 == ridx)
									costs[gid0 * num_outputs + ridx] = fma(0.9f, all_actions[gid0 * num_outputs + ridx], rewards[gid0]);
							}
						}

	__kernel void loss(__global const float *actions, __global float *costs, const size_t elements)
	{
		const size_t gid = get_global_id(0);
		if (gid < elements)
		{
			costs[gid] = pow(costs[gid] - actions[gid], 2.0f);
		}
	}
)";