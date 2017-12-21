cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(LiveSegmentation CXX)

set(LIVESEGMENTATION_VERSION_MAJOR "1")
set(LIVESEGMENTATION_VERSION_MINOR "0")
set(LIVESEGMENTATION_VERSION_PATCH "1")
set(LIVESEGMENTATION_VERSION "${LIVESEGMENTATION_VERSION_MAJOR}.${LIVESEGMENTATION_VERSION_MINOR}.${LIVESEGMENTATION_VERSION_PATCH}")

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall")

set(LiveSegmentaion_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

set(Caffe_DIR "/opt/caffe/build")

find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(LiveSegmentation STATIC ${SOURCES})

target_include_directories(LiveSegmentation PUBLIC $<BUILD_INTERFACE:${LiveSegmentaion_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
target_include_directories(LiveSegmentation PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(LiveSegmentation PRIVATE ${Caffe_DIR}/include)

target_link_libraries(LiveSegmentation ${OpenCV_LIBS})
target_link_libraries(LiveSegmentation ${Caffe_LIBRARIES})
target_link_libraries(LiveSegmentation -lpthread -lboost_system -lglog)


set(INCLUDE_INSTALL_ROOT_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
set(INSTALL_CMAKE_DIR ${LIB_INSTALL_DIR}/cmake/LiveSegmentation)

set_target_properties(LiveSegmentation PROPERTIES
	VERSION "${LIVESEGMENTATION_VERSION}"
	SOVERSION "${LIVESEGMENTATION_VERSION_MAJOR}.${LIVESEGMENTATION_VERSION_MINOR}"
)


install(TARGETS LiveSegmentation
        EXPORT LiveSegmentationTargets
        ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
        LIBRARY DESTINATION ${LIB_INSTALL_DIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
install(DIRECTORY ${LiveSegmentaion_INCLUDE_DIR}/LiveSegmentation
        DESTINATION ${INCLUDE_INSTALL_ROOT_DIR}
)



export(TARGETS LiveSegmentation FILE "${PROJECT_BINARY_DIR}/LiveSegmentationTargets.cmake")
export(PACKAGE LiveSegmentation)
set(EXPORT_TARGETS LiveSegmentation CACHE INTERNAL "export targets")

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INCLUDE_INSTALL_ROOT_DIR}")

set(CONFIG_INCLUDE_DIRS "\${LIVESEGMENTATION_SOURCE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfig.cmake.in
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfig.cmake" @ONLY)

set(CONFIG_INCLUDE_DIRS "\${LIVESEGMENTATION_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfig.cmake.in
	"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LiveSegmentationConfig.cmake" @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LiveSegmentationConfigVersion.cmake.in
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfigVersion.cmake" @ONLY)

install(FILES
	"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LiveSegmentationConfig.cmake"
	"${PROJECT_BINARY_DIR}/LiveSegmentationConfigVersion.cmake"
	DESTINATION "${INSTALL_CMAKE_DIR}")
install(EXPORT LiveSegmentationTargets DESTINATION ${INSTALL_CMAKE_DIR})
