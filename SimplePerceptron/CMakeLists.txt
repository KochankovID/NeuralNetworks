cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(SimplePerceptron ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../SimplePerceptron)

target_include_directories(SimplePerceptron PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_SimplePerceptron ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../SimplePerceptron/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(SimplePerceptron copy_target_SimplePerceptron)