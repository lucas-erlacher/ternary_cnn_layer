# This cmakelists works on ARM CPU with GCC
# Tested environment: Rpi 400, Rpi OS, cmake=3.25, GCC=12.2.0 
# It should work for similar GNU/Lunix based OS

cmake_minimum_required(VERSION 3.11.4) 
project(ASL_TAB)

set(CMAKE_C_STANDARD 11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flax-vector-conversions -march=armv8-a+simd -funsafe-math-optimizations -fbuiltin -O3")
# Reference cmake flags for X86_64 CPU
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse4.1 -msse4.2 -mavx2 -mpopcnt -march=core-avx2 -O3")

# file(GLOB_RECURSE main_sources TAB/*.cpp TAB/*.h main.cpp)
# file(GLOB_RECURSE verify_sources TAB/*.cpp TAB/*.h verify.cpp)
# file(GLOB measure_multi_sources measurement/tsc_x86.h measurement/measurement.h measurement/measurement.cpp
# optimization/*.cpp optimization/*.h TAB/*.cpp TAB/*.h measurement/measure_multi.cpp measurement/strategies.cpp)
file(GLOB measure_single_sources measurement/tsc_x86.h measurement/measurement.h measurement/measurement.cpp
optimization/*.cpp optimization/*.h TAB/*.cpp TAB/*.h measurement/measure_single.cpp measurement/strategies.cpp)
# file(GLOB_RECURSE gemm_sources TAB/common.h optimization/*.cpp optimization/*.h measurement/measure_GEMM.cpp)

# add_executable(main ${main_sources})
# add_executable(verify ${verify_sources})
# add_executable(measure ${measure_multi_sources})
add_executable(measure ${measure_single_sources})
# add_executable(gemm ${gemm_sources})