cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(NotSoSimplePerceptron_Sigm ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../NotSoSimplePerceptron_Sigm)

target_include_directories(NotSoSimplePerceptron_Sigm PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_NotSoSimplePerceptron_Sigm ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../NotSoSimplePerceptron_Sigm/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(NotSoSimplePerceptron_Sigm copy_target_NotSoSimplePerceptron_Sigm)