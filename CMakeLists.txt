cmake_minimum_required(VERSION 3.1.0)
project(CameraCalibration)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Protobuf REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(OpenCV 3 REQUIRED COMPONENTS viz core imgproc imgcodecs calib3d flann xfeatures2d)
find_package(PCL 1.8 REQUIRED COMPONENTS common search io filters)
find_package(Boost REQUIRED COMPONENTS program_options)



set(GRPC_LIBRARIES grpc++_unsecure grpc gpr)

set(INCLUDE_DIRS
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Quick_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
)

set(LIBRARIES
    ${PROTOBUF_LIBRARY}
    ${GRPC_LIBRARIES}
    ${Qt5Core_LIBRARIES}
    ${Qt5Gui_LIBRARIES}
    ${Qt5Widgets_LIBRARIES}
    ${Qt5Quick_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)

set(CameraCalibration_SOURCE_DIR "${PROJECT_SOURCE_DIR}")

include_directories(SYSTEM ${INCLUDE_DIRS}) # to avoid warnings from other libraries
include_directories(${CameraCalibration_SOURCE_DIR})

# GRPC
set(GRPC_CPP_PLUGIN_PATH "/usr/local/bin/grpc_cpp_plugin")
set(GRPC_PATH "${PROJECT_SOURCE_DIR}/grpc")
set(GRPC_FILES "${GRPC_PATH}/*.proto")

set(GENERATED_GRPC_PATH "${CMAKE_BINARY_DIR}/generated/grpc")
file(MAKE_DIRECTORY ${GENERATED_GRPC_PATH})
include_directories(SYSTEM ${GENERATED_GRPC_PATH})
set(GENERATED_GRPC_SOURCES
	"${GENERATED_GRPC_PATH}/CalibrationService.pb.h"
	"${GENERATED_GRPC_PATH}/CalibrationService.pb.cc"
	"${GENERATED_GRPC_PATH}/CalibrationService.grpc.pb.h"
	"${GENERATED_GRPC_PATH}/CalibrationService.grpc.pb.cc")

set_source_files_properties(${GENERATED_GRPC_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

add_custom_command(
    OUTPUT ${GENERATED_GRPC_SOURCES}
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS "--proto_path=${PROTO_PATH}"
         "--proto_path=${GRPC_PATH}"
         "--grpc_out=${GENERATED_GRPC_PATH}"
         "--plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_PATH}"
         "${GRPC_FILES}"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS "--proto_path=${PROTO_PATH}"
         "--proto_path=${GRPC_PATH}"
         "--cpp_out=${GENERATED_GRPC_PATH}"
         "${GRPC_FILES}"
    )

set(SOURCES
	"${GENERATED_GRPC_PATH}/CalibrationService.pb.cc"
	"${GENERATED_GRPC_PATH}/CalibrationService.grpc.pb.cc"
	"${CameraCalibration_SOURCE_DIR}/calibrationServer/CalibrationServer.cpp"
	"${CameraCalibration_SOURCE_DIR}/main.cpp")





add_executable(calibrationServer ${SOURCES})

target_link_libraries(calibrationServer ${LIBRARIES})

set_property(TARGET calibrationServer PROPERTY CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
