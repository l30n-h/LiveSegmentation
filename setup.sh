#!bin/sh

# Delete directories to allow a clean build
cmake -E cmake_echo_color --red ">>>>> Delete directories from old build"
cmake -E remove_directory bin
cmake -E remove_directory build

# Create directories
cmake -E cmake_echo_color --green ">>>>> Create directories from current build"
cmake -E make_directory bin
cmake -E make_directory build


# Build LiveSegmentation
cmake -E cmake_echo_color --blue ">>>>> Build LiveSegmentation"
cmake -E make_directory build/lib/LiveSegmentation
cd build/lib/LiveSegmentation
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../../bin ../../../lib/LiveSegmentation
make -j 13
make install
cd ../../..

# Invoke CMake on project
cmake -E cmake_echo_color --blue ">>>>> Build LiveSegmentationTest project"
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../bin ..
#cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../bin ..
cd ..


# Build project
sh build.sh

