cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(P_Network_one ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH .)

target_include_directories(P_Network_one PRIVATE ${ANN_INCLUDE_DIR}
        ../lib/fast-cpp-csv-parser)

add_custom_target(copy_target_P_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources resources
        COMMENT "Copy neded folder to compiling executable")
add_dependencies(P_Network_one copy_target_P_Network)