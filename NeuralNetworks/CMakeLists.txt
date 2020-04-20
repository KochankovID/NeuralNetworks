cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
                                    # Если версия установленой программы
                                    # старее указаной, произайдёт аварийный выход.
project(NeuralNetworks)                # Название проекта
file(GLOB_RECURSE TARGET_SRC "./src/*.h")
add_library(NeuralNetworks ${TARGET_SRC} src/ANN.h src/Instruments.h src/CNN.h)

set(directory
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron/Weights
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/LearnNeyron
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/Filter
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/LearnFilter
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Functors
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/GradDence
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Initializers
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Matrix
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Metrics
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Tensor
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Instruments/Vector
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/ConvolutionLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/DenceLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/FlattenLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/Layer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/MaxpoolingLayer
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model
        ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Preprocessing
        )

file(MAKE_DIRECTORY ${directory})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Neyron.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Weights.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron/Weights)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/LearnNeyron.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/LearnNeyron)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Filter.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/Filter)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/LearnFilter.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/LearnFilter)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Func.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Functors)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Functors.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Functors)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Grad.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/GradDence)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Gradients.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/GradDence)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Init.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Initializers)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Initializers.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Initializers)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Matrix)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Metr.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Metrics)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Metrics.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Metrics)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Tensor.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Tensor)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Vector.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Vector)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/ConvolutionLayer.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/ConvolutionLayer)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/DenceLayer.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/DenceLayer)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/FlattenLayer.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/FlattenLayer)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/Layer)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/MaxpoolingLayer.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/MaxpoolingLayer)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Model.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Data.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Preprocessing)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/ANN.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/CNN.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments)

