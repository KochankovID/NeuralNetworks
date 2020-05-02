cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
                                    # Если версия установленой программы
                                    # старее указаной, произайдёт аварийный выход.
project(NeuralNetworks)                # Название проекта

file(GLOB_RECURSE TARGET_SRC "./src/*.cpp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib)
add_library(NeuralNetworks ${TARGET_SRC})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/headers/NN DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/lib DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../neuralnetworks-config.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../)
