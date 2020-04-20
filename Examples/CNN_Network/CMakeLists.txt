cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(CNN_Network ${TARGET_SRC})
