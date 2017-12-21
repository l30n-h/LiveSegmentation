cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(LiveSegmentation CXX)

set(LIVESEGMENTATION_VERSION_MAJOR "1")
set(LIVESEGMENTATION_VERSION_MINOR "0")
set(LIVESEGMENTATION_VERSION_PATCH "1")
set(LIVESEGMENTATION_VERSION "${LIVESEGMENTATION_VERSION_MAJOR}.${LIVESEGMENTATION_VERSION_MINOR}.${LIVESEGMENTATION_VERSION_PATCH}")

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall")

set(Caffe_DIR "/opt/caffe/build/")

find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

get_property(Caffe_INCLUDE_DIRS TARGET caffe PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND Caffe_INCLUDE_DIRS ${Caffe_DIR}include)

get_property(Caffe_COMPILE_DEFINITIONS TARGET caffe PROPERTY INTERFACE_COMPILE_DEFINITIONS)

message(STATUS "Caffe_COMPILE_DEFINITIONS")
message(STATUS ${Caffe_COMPILE_DEFINITIONS})

message(STATUS "Caffe_INCLUDE_DIRS")
message(STATUS ${Caffe_INCLUDE_DIRS})

message(STATUS "Caffe_LIBRARIES")
message(STATUS ${Caffe_LIBRARIES})



file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")



set(INCLUDE_INSTALL_ROOT_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_ROOT_DIR}/LiveSegmentation)
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
set(_INSTALL_DESTINATIONS
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}"
)
set(INSTALL_CMAKE_DIR ${LIB_INSTALL_DIR}/cmake/LiveSegmentation)

add_library(LiveSegmentation STATIC ${SOURCES})


target_include_directories(LiveSegmentation PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(LiveSegmentation PRIVATE ${Caffe_INCLUDE_DIRS})

target_compile_definitions(LiveSegmentation PUBLIC ${Caffe_COMPILE_DEFINITIONS})

target_link_libraries(LiveSegmentation ${OpenCV_LIBS})
#target_link_libraries(LiveSegmentation {Caffe_LIBRARIES})
target_link_libraries(LiveSegmentation ${Caffe_DIR}lib/libcaffe.so -lglog -lprotobuf -lboost_system -lpthread)


#add_executable(nyu_classification src/nyu_classification.cpp)
#target_link_libraries(nyu_classification LiveSegmentation ${OpenCV_LIBS})

#include_directories(${Caffe_INCLUDE_DIRS})
#add_executable( nyu_classification nyu_classification.cpp BA/Classifier.cpp BA/ClassifierQueue.cpp )
#target_link_libraries( nyu_classification ${OpenCV_LIBS} ${Caffe_LIBRARIES})


set_target_properties(LiveSegmentation PROPERTIES
	VERSION "${LIVESEGMENTATION_VERSION}"
	SOVERSION "${LIVESEGMENTATION_VERSION_MAJOR}.${LIVESEGMENTATION_VERSION_MINOR}"
)

install(TARGETS LiveSegmentation EXPORT LiveSegmentationTargets ${_INSTALL_DESTINATIONS})
install(
    DIRECTORY "include/LiveSegmentation/"
    DESTINATION ${INCLUDE_INSTALL_DIR}
)



export(
    TARGETS LiveSegmentation
    FILE "${PROJECT_BINARY_DIR}/LiveSegmentationTargets.cmake")
export(PACKAGE LiveSegmentation)
set(EXPORT_TARGETS LiveSegmentation CACHE INTERNAL "export targets")

set(CONFIG_INCLUDE_DIRS "\${LIVESEGMENTATION_SOURCE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfig.cmake.in
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfig.cmake" @ONLY)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INCLUDE_INSTALL_ROOT_DIR}")
set(CONFIG_INCLUDE_DIRS "\${LIVESEGMENTATION_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfig.cmake.in
	"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LiveSegmentationConfig.cmake" @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfigVersion.cmake.in
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfigVersion.cmake" @ONLY)

install(FILES
	"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LiveSegmentationConfig.cmake"
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfigVersion.cmake"
	DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
install(EXPORT LiveSegmentationTargets DESTINATION ${INSTALL_CMAKE_DIR})
