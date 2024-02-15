cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX_CPP_TEST)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TEST_NAME ipex_cpp_test)

# specify the C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#Custome Setting, can be set thought cmake parameter
set(PYTORCH_INSTALL_DIR /home/lesliefang/pytorch_1_7_1/runtime/pytorch-spr/torch)
set(IPEX_INSTALL_DIR /home/lesliefang/pytorch_1_7_1/runtime/frameworks.ai.pytorch.ipex-cpu/build/Release/packages/intel_extension_for_pytorch)
#set(IPEX_INSTALL_DIR /home/lesliefang/pytorch_1_7_1/runtime/frameworks.ai.pytorch.ipex-cpu/build/Debug/packages/intel_extension_for_pytorch)
set(IPEX_INCLUDE_DIR /home/lesliefang/pytorch_1_7_1/runtime/frameworks.ai.pytorch.ipex-cpu/intel_extension_for_pytorch)

# Set the include dir
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include/)
include_directories(${IPEX_INCLUDE_DIR})

# Add the Test Files
set(IPEX_CPP_TEST_SOURCES main.cpp)

add_executable(${TEST_NAME} ${IPEX_CPP_TEST_SOURCES})

# Link Pytorch
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${TEST_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)

# Link IPEX
target_link_libraries(${TEST_NAME} PUBLIC ${IPEX_INSTALL_DIR}/lib/libintel-ext-pt-cpu.so)

