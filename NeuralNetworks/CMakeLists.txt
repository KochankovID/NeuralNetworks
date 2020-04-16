cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
									# Если версия установленой программы
									# старее указаной, произайдёт аварийный выход.
project(ArtificialNN)				# Название проекта

add_custom_target(copy_target ALL
${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ./include
COMMENT "Copy neded folder to compiling executable")
