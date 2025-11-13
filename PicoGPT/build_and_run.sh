#!/bin/bash

# Build and run script for PicoGPT
# Exits if build fails

set -e  # Exit on any error

echo "Building PicoGPT..."

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building project..."
make

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Running PicoGPT..."
    echo "=========================================="
    ./PicoGPT
else
    echo "Build failed!"
    exit 1
fi