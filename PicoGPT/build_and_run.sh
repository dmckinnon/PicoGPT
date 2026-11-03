#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "$0")"

# The Makefile builds native programs; CMake builds Pico firmware.
make gpt_model
exec ./gpt_model