cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(CNN_Network ${TARGET_SRC})

set(EXECUTABLE_OUTPUT_PATH  ../CNN_Network)

target_include_directories(CNN_Network PRIVATE ${ANN_INCLUDE_DIR})

add_custom_target(copy_target_CNN_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../CNN_Network/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(CNN_Network copy_target_CNN_Network)