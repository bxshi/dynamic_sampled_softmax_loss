cmake_minimum_required(VERSION 3.0)
project(dynamic_sampled_softmax_loss)

set(CMAKE_CXX_STANDARD 11)

EXEC_PROGRAM(uname OUTPUT_VARIABLE OS)
MESSAGE(STATUS "OS Type: " ${OS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared -fPIC -O2")

if (${OS} STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
endif ()

EXEC_PROGRAM(python3 ARGS "-c 'import tensorflow as tf; print(tf.sysconfig.get_include())'" OUTPUT_VARIABLE TF_FILES)
message(STATUS "TensorFlow include dir: " ${TF_FILES})

set(SOURCE_FILES dynamic_candidate_sampler_ops.cc dynamic_range_sampler.cc dynamic_range_sampler.h dynamic_candidate_sampling_ops.cc)

include_directories(${TF_FILES})

add_library(dynamic_sampled_softmax_loss SHARED ${SOURCE_FILES})
