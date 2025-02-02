# CLRL

## What is it?

This project implements a basic Deep Q-Learning model using OpenCL and C++. This enables cross platform portability while using the gpu for more speed.

## Build instructions

Building is very simple with no macros to be specified other than the optional install prefix. Example: 

  1. `cmake -B CLRL/build -S CLRL -D CMAKE_INSTALL_PREFIX=CLRL`
  2. `cmake --build CLRL/build --target install`