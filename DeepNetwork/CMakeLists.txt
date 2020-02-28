cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(DeepNetwork ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH  ../DeepNetwork)


target_include_directories(DeepNetwork PRIVATE ${ANN_INCLUDE_DIR})

target_include_directories(DeepNetwork PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/ml/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "${CMAKE_BINARY_DIR}"
        )

target_link_libraries(DeepNetwork opencv_core opencv_imgcodecs)

add_custom_target(copy_target_DeepNetwork ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../DeepNetwork/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(DeepNetwork copy_target_DeepNetwork)