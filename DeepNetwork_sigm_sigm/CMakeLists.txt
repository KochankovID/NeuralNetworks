cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(DeepNetwork_sigm_sigm ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH .)


target_include_directories(DeepNetwork_sigm_sigm PRIVATE ${ANN_INCLUDE_DIR})

target_include_directories(DeepNetwork_sigm_sigm PRIVATE ${ANN_INCLUDE_DIR}
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

target_link_libraries(DeepNetwork_sigm_sigm opencv_core opencv_imgcodecs)

add_custom_target(copy_target_DeepNetwork_sigm_sigm ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(DeepNetwork_sigm_sigm copy_target_DeepNetwork_sigm_sigm)