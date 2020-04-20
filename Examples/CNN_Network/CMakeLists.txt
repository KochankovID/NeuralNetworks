cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(CNN_Network ${TARGET_SRC})

set(EXECUTABLE_OUTPUT_PATH  ../CNN_Network)
find_package (Boost COMPONENTS filesystem REQUIRED)

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
        "${Boost_INCLUDE_DIRS}"
        )

target_link_libraries(CNN_Network ${Boost_LIBRARIES})
target_link_libraries(CNN_Network opencv_core opencv_imgcodecs)

add_custom_target(copy_target_CNN_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources ../CNN_Network/resources
        COMMENT "Copy neded folder to compiling executable")

add_dependencies(CNN_Network copy_target_CNN_Network)