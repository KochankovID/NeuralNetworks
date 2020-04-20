cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(DFF_Network ${TARGET_SRC})

add_custom_target(copy_target_DFF_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(DFF_Network copy_target_DFF_Network)