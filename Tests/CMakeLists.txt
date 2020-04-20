cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Test ${TARGET_SRC} src/Test_DenceLayer/test_DenceLayer.cpp src/Test_LearnFilter/test_LearnFilter.cpp src/Test_Tensor/test_Tensor.cpp src/Test_ConvolutionLayer/test_ConvolutionLayer.cpp src/Test_FlattenLayer/test_FlattenLayer.cpp src/Test_MaxpoolingLayer/test_MaxpoolingLayer.cpp)

enable_testing()

find_package(GTest CONFIG REQUIRED)
target_link_libraries(Test PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

add_test(AllTestsInMain Test)

set(EXECUTABLE_OUTPUT_PATH  ../Tests)

target_include_directories(Test PRIVATE ${ANN_INCLUDE_DIR})