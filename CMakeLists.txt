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

set(NeuralNetworks_INCLUDE_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron/Weights
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/LearnNeyron
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/Filter
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/LearnFilter
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Functors
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/GradDence
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Initializers
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Matrix
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Metrics
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Tensor
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Vector
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/ConvolutionLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/DenceLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/FlattenLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/Layer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/MaxpoolingLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Preprocessing
        ${CMAKE_CURRENT_BINARY_DIR}/../lib/fast-cpp-csv-parser
        )

set(NeuralNetworks_SOURCE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/ANN/Neyron
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/ANN/Neyron/Weights
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/ANN/LearnNeyron
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/CNN/Filter
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/CNN/LearnFilter
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Functors
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/GradDence
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Initializers
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Matrix
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Metrics
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments/Vector
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model/ConvolutionLayer
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model/DenceLayer
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model/FlattenLayer
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model/Layer
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model/MaxpoolingLayer
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/Model
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Preprocessing
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/ANN
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/SU/CNN
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/src/NN/Instruments
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/lib/fast-cpp-csv-parser
        )

include_directories(${NeuralNetworks_SOURCE_DIR})

find_package (Boost COMPONENTS filesystem REQUIRED)
link_libraries(${Boost_LIBRARIES})
include_directories("${Boost_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS})

add_subdirectory(NeuralNetworks)
add_subdirectory(Examples)
add_subdirectory(Tests)
