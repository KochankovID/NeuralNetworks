cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
                                    # Если версия установленой программы
                                    # старее указаной, произайдёт аварийный выход.
project(ArtificialNN)                # Название проекта
add_library(NeuralNetworks INTERFACE)
file(GLOB TARGET_SRC "./src/*.h" )
target_include_directories(NeuralNetworks PRIVATE ${TARGET_SRC})

add_custom_target(copy_target ALL
${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/../include/NN
COMMENT "Copy neded folder to compiling executable")
