cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(NotSoSimplePerceptron ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../NotSoSimplePerceptron)

target_include_directories(NotSoSimplePerceptron PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_NotSoSimplePerceptron ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../NotSoSimplePerceptron/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(NotSoSimplePerceptron copy_target_NotSoSimplePerceptron)