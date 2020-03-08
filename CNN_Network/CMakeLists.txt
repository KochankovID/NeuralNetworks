cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(CNN_Network ${TARGET_SRC} ../ArtificialNN/src/Neyron/Neyrons.h ../ArtificialNN/src/Neyron/LearnNeyron/LearnNeyron.h ../ArtificialNN/src/Instruments/GradDence/Grad.h ../ArtificialNN/src/Instruments/GradDence/Gradients.h ../ArtificialNN/src/Instruments/Metrix/Metr.h ../ArtificialNN/src/Instruments/Metrix/Metrix.h ../ArtificialNN/src/Instruments/Matrix/Matrices.h ../ArtificialNN/src/Neyron/DenceNeyron.h)

set(EXECUTABLE_OUTPUT_PATH  ../CNN_Network)

target_include_directories(CNN_Network PRIVATE ${ANN_INCLUDE_DIR})

target_include_directories(CNN_Network PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "${CMAKE_BINARY_DIR}"
        )

target_link_libraries(CNN_Network opencv_core opencv_imgcodecs)

add_custom_target(copy_target_CNN_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../CNN_Network/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(CNN_Network copy_target_CNN_Network)