cmake_minimum_required(VERSION 3.0.0)

file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(FF_Network ${TARGET_SRC})
set(EXECUTABLE_OUTPUT_PATH .)

target_include_directories(FF_Network PRIVATE ${ANN_INCLUDE_DIR})
find_package (Boost COMPONENTS filesystem REQUIRED)

add_custom_target(copy_target_FF_Network ALL
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/resources resources
        COMMENT "Copy neded folder to compiling executable")

target_include_directories(FF_Network PRIVATE ${ANN_INCLUDE_DIR}
        "../lib/opencv/include"
        "../lib/opencv/modules/core/include"
        "../lib/opencv/modules/ts/include"
        "../lib/opencv/modules/highgui/include"
        "../lib/opencv/modules/imgcodecs/include"
        "../lib/opencv/modules/videoio/include"
        "../lib/opencv/modules/imgproc/include"
        "../lib/opencv/modules"
        "../lib/fast-cpp-csv-parser"
        "${CMAKE_BINARY_DIR}"
        "${Boost_INCLUDE_DIRS}"
        )

target_link_libraries(FF_Network opencv_core opencv_imgcodecs)
target_link_libraries(FF_Network ${Boost_LIBRARIES})

add_dependencies(FF_Network copy_target_FF_Network)