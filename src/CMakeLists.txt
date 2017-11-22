cmake_minimum_required(VERSION 2.8.7)
project( nyu_classification CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall")

set(Caffe_DIR "/opt/caffe/build")

#set(Caffe_INCLUDE_DIRS "/opt/caffe/build/install/include" "/opt/caffe/include")
set(Caffe_INCLUDE_DIRS "/opt/caffe/build/install/include")
set(Caffe_INCLUDE_DIRS "/opt/caffe/build/include")
#set(Caffe_LIBRARIES "/opt/caffe/build/install/lib/libcaffe.so")

find_package( OpenCV REQUIRED )
find_package( Caffe REQUIRED )

message(STATUS "Caffe_DEFINITIONS")
message(STATUS ${Caffe_DEFINITIONS})

message(STATUS "Caffe_INCLUDE_DIRS")
message(STATUS ${Caffe_INCLUDE_DIRS})

message(STATUS "Caffe_LIBRARIES")
message(STATUS ${Caffe_LIBRARIES})

include_directories(${Caffe_INCLUDE_DIRS})
#add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable( nyu_classification nyu_classification.cpp BA/Classifier.cpp BA/ClassifierQueue.cpp )
target_link_libraries( nyu_classification ${OpenCV_LIBS} ${Caffe_LIBRARIES})
