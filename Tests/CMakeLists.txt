cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

add_executable(Test ${TARGET_SRC})

enable_testing()

add_subdirectory(lib/googletest)

target_link_libraries(Test PRIVATE gtest gtest_main pthread NeuralNetworks)
target_include_directories(Test PRIVATE ${GTEST_INCLUDE_DIR})

add_test(AllTestsInMain Test)

set(EXECUTABLE_OUTPUT_PATH  ../Tests)