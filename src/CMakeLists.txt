cmake_minimum_required(VERSION ${MINIMUM_CMAKE_VERSION} FATAL_ERROR)


project(src)


find_package(OpenCV REQUIRED)
find_package(LiveSegmentation 1.0.1 REQUIRED)

include_directories(${LIVESEGMENTATION_INCLUDE_DIRS})


add_executable(nyu_classification nyu_classification.cpp)
target_link_libraries(nyu_classification ${LIVESEGMENTATION_LIBRARIES} ${OpenCV_LIBS})

add_executable(map_test map_test.cpp)
target_link_libraries(map_test ${LIVESEGMENTATION_LIBRARIES} ${OpenCV_LIBS})
