set(MINIMUM_CMAKE_VERSION "2.8.12")
cmake_minimum_required(VERSION ${MINIMUM_CMAKE_VERSION} FATAL_ERROR)

# The name of the project
project(LiveSegmentationTest)


if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(DEBUG_MODE 1)
    message("+------------------------------------------------------------------------------+")
    message("|                                                                              |")
    message("|                            Building in DEBUG MODE                            |")
    message("|                                                                              |")
    message("+------------------------------------------------------------------------------+")
elseif(${CMAKE_BUILD_TYPE} MATCHES Release)
    message("+------------------------------------------------------------------------------+")
    message("|                                                                              |")
    message("|                           Building in RELEASE MODE                           |")
    message("|                                                                              |")
    message("+------------------------------------------------------------------------------+")
else()
    message(FATAL_ERROR "Build Type not specified")
endif()


# The project directory structure
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_BIN_DIR   "${PROJECT_ROOT_DIR}/bin")
set(PROJECT_BUILD_DIR "${PROJECT_ROOT_DIR}/build")
set(PROJECT_LIB_DIR   "${PROJECT_ROOT_DIR}/lib")

# The output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BIN_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BIN_DIR})



# Add own module path to CMake
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/CMake/")
message(STATUS "Added custom module path : ${CMAKE_MODULE_PATH}")



# Find CUDA beforehand
find_package(CUDA 8.0 )



# Separate Compilation for CUDA files, i.e. using a linker for device code
# - NOTE : This is disabled due to an unresolved bug in CMake
#set(CUDA_SEPARABLE_COMPILATION ON)



# Device flags (.cu files)
set(SUPPORTED_CUDA_ARCHS 35 50 52 61)
foreach(ARCH IN LISTS SUPPORTED_CUDA_ARCHS)            
    list(APPEND COMPILATION_FLAGS "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    message(STATUS "Enabled compilation for CC ${ARCH}")
endforeach()
list(APPEND COMPILATION_FLAGS "-std=c++11")
list(APPEND COMPILATION_FLAGS "-Xcompiler;-Wall")
list(APPEND COMPILATION_FLAGS "-Xcompiler;-Wextra")

list(APPEND OPTIMIZATION_FLAGS "-O3")
list(APPEND OPTIMIZATION_FLAGS "-use_fast_math")

list(APPEND WARNING_SUPPRESSION_FLAGS "--expt-relaxed-constexpr")       # Eigen 3.3+

list(APPEND DEVICE_FLAGS ${COMPILATION_FLAGS})
list(APPEND DEVICE_FLAGS ${OPTIMIZATION_FLAGS})
list(APPEND DEVICE_FLAGS ${WARNING_SUPPRESSION_FLAGS})
message(STATUS "Created device flags : ${DEVICE_FLAGS}")


# Host flags (.cpp files)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND HOST_FLAGS "-std=c++14")
list(APPEND HOST_FLAGS "-Wall")
list(APPEND HOST_FLAGS "-fPIC")
#list(APPEND HOST_FLAGS "-pedantic")
#list(APPEND HOST_FLAGS "-Wextra")
#list(APPEND HOST_FLAGS "-Wconversion")          # Only add this here since thrust has problem with this
#list(APPEND HOST_FLAGS "-Wsign-conversion")     # Only add this here since thrust has problem with this
#list(APPEND HOST_FLAGS "-O3")
message(STATUS "Created  host  flags : ${HOST_FLAGS}")


# Apply compiler flags
list(APPEND CUDA_NVCC_FLAGS ${DEVICE_FLAGS})
list(APPEND CMAKE_CXX_FLAGS ${HOST_FLAGS})
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")        # CMAKE_CXX_FLAGS must be separated by spaces rather than semicolons
message(STATUS "Building with NVCC flags : ${CUDA_NVCC_FLAGS}")
message(STATUS "Building with CXX  flags : ${CMAKE_CXX_FLAGS}")



# Manually set the version information until CMake 3.0+ can be used
set(LiveSegmentationTest_VERSION_MAJOR 1)
set(LiveSegmentationTest_VERSION_MINOR 0)
set(LiveSegmentationTest_VERSION_PATCH 0)
set(LiveSegmentationTest_VERSION "${LiveSegmentationTest_VERSION_MAJOR}.${LiveSegmentationTest_VERSION_MINOR}.${LiveSegmentationTest_VERSION_PATCH}")



# Local project structure
set(LOCAL_LOG_DIRECTORY             "${PROJECT_BIN_DIR}/log")

file(MAKE_DIRECTORY ${LOCAL_LOG_DIRECTORY})

set(LOCAL_PROJECT_PATHS_CONFIG_DIR  "${PROJECT_BIN_DIR}")
configure_file("${PROJECT_ROOT_DIR}/LocalProjectPaths.h.in"
               "${LOCAL_PROJECT_PATHS_CONFIG_DIR}/LocalProjectPaths.h")
set(LOCAL_PROJECT_PATHS_INCLUDE_DIR "${LOCAL_PROJECT_PATHS_CONFIG_DIR}")



# CTest must be enabled from top CMakeLists.txt
#enable_testing()

# Build project
add_subdirectory("src")
