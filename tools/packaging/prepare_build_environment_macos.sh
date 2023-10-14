#! /bin/bash

set -e
set -x

echo "CUDA is not available on MacOS"

echo "Installing libomp"
brew install llvm libomp
