cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetworks VERSION 0.1.0)

set(NeuralNetworks_DIR ${NeuralNetworks_DIR} "${CMAKE_CURRENT_BINARY_DIR}/include")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
	message(-- Building <ArtificialNN> for <${CMAKE_CXX_COMPILER_ID}> compiler)
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:SSE /arch:SSE2 /arch:AVX")
	SET(BUILD_SHARED_LIBS OFF)

elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL AppleClang)
	message(-- Building <ArtificialNN> for <${CMAKE_CXX_COMPILER_ID}> compiler)
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
	message(-- Building <ArtificialNN> for <${CMAKE_CXX_COMPILER_ID}> compiler)
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 --param ggc-min-expand=1 --param ggc-min-heapsize=32768")
else()
	message(-- Building <ArtificialNN> for ${CMAKE_CXX_COMPILER_ID} compiler)
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

find_package (Boost COMPONENTS filesystem REQUIRED)
link_libraries(${Boost_LIBRARIES})
include_directories("${Boost_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS})

add_subdirectory(NeuralNetworks)
add_subdirectory(Examples)
add_subdirectory(Tests)
