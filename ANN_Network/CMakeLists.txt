cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(DeepNetwork_relu_sigm ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH .)

find_package (Boost COMPONENTS filesystem REQUIRED)

target_include_directories(DeepNetwork_relu_sigm PRIVATE ${ANN_INCLUDE_DIR})

target_include_directories(DeepNetwork_relu_sigm PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/ml/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "../lib/fast-cpp-csv-parser"
        "${CMAKE_BINARY_DIR}"
        "${Boost_INCLUDE_DIRS}"
        )

target_link_libraries(DeepNetwork_relu_sigm opencv_core opencv_imgcodecs)
target_link_libraries(DeepNetwork_relu_sigm ${Boost_LIBRARIES})

add_custom_target(copy_target_DeepNetwork_relu_sigm ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../DeepNetwork_relu_sigm/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(DeepNetwork_relu_sigm copy_target_DeepNetwork_relu_sigm)