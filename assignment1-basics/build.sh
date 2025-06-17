#!/bin/bash

c++ -I$(pwd) -O3 -Wall -shared -std=c++17 -fPIC $(uv run python -m pybind11 --includes) bpe.cxx -o bpe$(uv run python3-config --extension-suffix)
c++ -I$(pwd) -O3 -Wall -shared -std=c++17 -fPIC $(uv run python -m pybind11 --includes) encoder.cxx -o encoder$(uv run python3-config --extension-suffix)
