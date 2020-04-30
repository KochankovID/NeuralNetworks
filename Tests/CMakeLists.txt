cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Test ${TARGET_SRC} src/Test_DenceLayer/test_DenceLayer.cpp src/Test_LearnFilter/test_LearnFilter.cpp src/Test_Tensor/test_Tensor.cpp src/Test_ConvolutionLayer/test_ConvolutionLayer.cpp src/Test_FlattenLayer/test_FlattenLayer.cpp src/Test_MaxpoolingLayer/test_MaxpoolingLayer.cpp src/Test_Ndarray/test_Ndarray.cpp ../NeuralNetworks/src/NN/UNSU/SOM/SOM.h ../NeuralNetworks/src/NN/UNSU/SOM/SOM.cpp)

enable_testing()

add_subdirectory(lib/gtest)



target_link_libraries(Test PRIVATE gtest gtest_main pthread)
target_include_directories(Test PRIVATE ${GTEST_INCLUDE_DIR})

add_test(AllTestsInMain Test)

set(EXECUTABLE_OUTPUT_PATH  ../Tests)