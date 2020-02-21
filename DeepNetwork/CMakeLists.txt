cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(DeepNetwork ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../DeepNetwork)


target_include_directories(DeepNetwork PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_DeepNetwork ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../DeepNetwork/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(DeepNetwork copy_target_DeepNetwork)