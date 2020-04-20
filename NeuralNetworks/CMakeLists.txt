cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
                                    # Если версия установленой программы
                                    # старее указаной, произайдёт аварийный выход.
project(NeuralNetworks)                # Название проекта

add_library(NeuralNetworks
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/ANN/Neyron/Neyron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/ANN/Neyron/Weights/Weights.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/ANN/LearnNeyron/LearnNeyron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/CNN/Filter/Filter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/CNN/LearnFilter/LearnFilter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Functors/Functors.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Functors/Func.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/GradDence/Grad.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/GradDence/Gradients.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Initializers/Init.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Initializers/Initializers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Matrix/Matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Metrics/Metr.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Metrics/Metrics.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Tensor/Tensor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Vector/Vector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/ConvolutionLayer/ConvolutionLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/DenceLayer/DenceLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/FlattenLayer/FlattenLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/Layer/Layer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/MaxpoolingLayer/MaxpoolingLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/Model/Model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Preprocessing/Data.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/ANN/ANN.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/SU/CNN/CNN.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NN/Instruments/Instruments.h
        )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/NN DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include)
#file(MAKE_DIRECTORY ${NeuralNetworks_INCLUDE_DIR})
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Neyron.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Weights.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/Neyron/Weights)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/LearnNeyron.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN/LearnNeyron)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Filter.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/Filter)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/LearnFilter.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN/LearnFilter)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Func.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Functors)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Functors.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Functors)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Grad.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/GradDence)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Gradients.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/GradDence)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Init.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Initializers)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Initializers.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Initializers)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Matrix)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Metr.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Metrics)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Metrics.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Metrics)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Tensor.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Tensor)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Vector.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments/Vector)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/ConvolutionLayer.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/ConvolutionLayer)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/DenceLayer.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/DenceLayer)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/FlattenLayer.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/FlattenLayer)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/Layer)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/MaxpoolingLayer.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model/MaxpoolingLayer)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Model.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/Model)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Data.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Preprocessing)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/ANN.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/ANN)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/CNN.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/SU/CNN)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments.h
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include/NN/Instruments)

