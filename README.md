# clRL

## What is it?

This project creates a basic RL model in OpenCL that uses the [clEnvironment](https://github.com/CodingWinner/clEnvironment) library for the environment.
It uses OpenCL to ensure cross platform portability and allows the user to leverage any compute device.

## Build instructions

The build has dependencies on [CMake](https://cmake.org/download/), [clEnvironment](https://github.com/CodingWinner/clEnvironment), [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK), and [CLBlast](https://github.com/CNugteren/CLBlast/tree/master).
Example build for a shared library without debug info:

	cmake -B build -S . -D CREATE_SHARED_LIB=ON -D CMAKE_INSTALL_PREFIX=.
	cmake --build build --target install --config Release