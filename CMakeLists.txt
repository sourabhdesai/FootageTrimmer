cmake_minimum_required(VERSION 3.3)
project(svdesai2_final)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -I/usr/local/include/opencv -lm -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc")

set(SOURCE_FILES
        FootageTrimmer.cpp
        FootageTrimmer.h
        main.cpp ImgDiffProcessor.cpp ImgDiffProcessor.h)

add_executable(svdesai2_final ${SOURCE_FILES})