cmake_minimum_required(VERSION 3.0.0)# Проверка версии CMake.
                                    # Если версия установленой программы
                                    # старее указаной, произайдёт аварийный выход.
project(ArtificialNN)                # Название проекта
add_library(NeuralNetworks
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ANN/Neyron/Neyron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ANN/Neyron/Weights/Weights.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ANN/LearnNeyron/LearnNeyron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CNN/Filter/Filter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CNN/LearnFilter/LearnFilter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Functors/Functors.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Functors/Func.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Functors/Func.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/GradDence/Gradients.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/GradDence/Grad.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Initializers/Initializers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Initializers/Init.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Matrix/Matrix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Metrics/Metrics.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Metrics/Metr.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Tensor/Tensor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Instruments/Vector/Vector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/ConvolutionLayer/ConvolutionLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/DenceLayer/DenceLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Flattenlayer/Flattenlayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Layer/Layer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/MaxpoolingLayer/MaxpoolingLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model/Model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Preprocessing/Data.h
        )


add_custom_target(copy_target ALL
${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/../include/NN
COMMENT "Copy neded folder to compiling executable")
