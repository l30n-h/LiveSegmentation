#!bin/sh

# Build project
cd build
# Number of jobs = 2 * cores + 1 due to Hyper-Threading
make -j 13 #VERBOSE=1
# Also build tests
make -j 13 buildtests
cd ..
