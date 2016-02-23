cmake_minimum_required(VERSION 2.8)

PROJECT (SteerableFilter)

find_package(OpenCV  REQUIRED)


file(GLOB SteerableFilter_SRCS
    "*.hpp"
    "*.cpp")
ADD_EXECUTABLE (SteerableFilter ${SteerableFilter_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( SteerableFilter ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

