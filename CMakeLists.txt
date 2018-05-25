# cmake_minimum_required(VERSION <specify CMake version here>)
project(MatMul__Success_)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)
include_directories(Cpp_common)

add_executable(MatMul__Success_
        Cpp_common/cl.hpp
        Cpp_common/device_picker.hpp
        Cpp_common/err_code.h
        Cpp_common/util.hpp
        Cpp_common/wtime.c
        matmul.cpp
        matmul.hpp
        matrix_lib.cpp
        matrix_lib.hpp)
