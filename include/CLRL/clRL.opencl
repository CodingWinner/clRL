std::string source = R"(

__kernel void leaky_relu(__global float *outputs, __global float *costs)
{
  const size_t gid = get_global_id(0);
  costs[gid] = 1.0f;
  if (outputs[gid] < 0)
  {
    costs[gid] = 0.01f;
    outputs[gid] *= 0.01f;
  }
}

__kernel void explore(__global uint *actions, const int seed, const uint max_val)
{
  if ((seed + get_global_id(0)) % 20 == 19)
  {
    int val = seed + get_global_id(0);
    val ^= (val >> 21);
    val ^= (val << 35);
    val ^= (val >> 4);
    actions[get_global_id(0)] = (val % (max_val + 1));
  }
}

__kernel void loss(__global const float *target_vals, __global const float *vals, __global const float *rewards, __global float *costs, __global uint *target_actions, __global uint *actions, const size_t batch_size)
{
  const size_t gid = get_global_id(0);
  costs[actions[gid] * batch_size + gid] = pown(fma(0.9f, target_vals[target_actions[gid] * batch_size + gid], rewards[gid]) - vals[actions[gid] * batch_size + gid], 2);
}

)";