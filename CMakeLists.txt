cmake_minimum_required(VERSION 3.6)
project(dynamic_sampled_softmax_loss)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared -undefined dynamic_lookup -fPIC -O2")

set(SOURCE_FILES dynamic_candidate_sampler_ops.cc dynamic_range_sampler.cc dynamic_range_sampler.h dynamic_candidate_sampling_ops.cc)
set(TF_FILES /Users/bshi/Code/py34env/lib/python3.4/site-packages/tensorflow/include)
include_directories(${TF_FILES})
add_library(dynamic_sampled_softmax_loss SHARED ${SOURCE_FILES})