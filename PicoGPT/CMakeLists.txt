# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
cmake_minimum_required(VERSION 3.17)

set(PICO_BOARD pico2 CACHE STRING "Target board")
set(PICO_PLATFORM rp2350-arm-s CACHE STRING "Target platform")
include(pico_sdk_import.cmake)

project(PicoGPT C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

pico_sdk_init()

if(NOT PICO_PLATFORM STREQUAL "rp2350-arm-s")
    message(FATAL_ERROR "This bring-up target requires the RP2350 Arm cores.")
endif()

option(GPT_MATMUL_UNROLL4 "Use four-output matmul in model inference" OFF)

add_executable(PicoGPT pico_main.c pico_memory.c gpt_vocab.c)
target_include_directories(PicoGPT PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(PicoGPT pico_stdlib hardware_clocks m)
target_compile_options(PicoGPT PRIVATE
    -Wall -Wextra -O2 -fstack-usage -ffp-contract=off
)
target_compile_definitions(PicoGPT PRIVATE
    PICO_STACK_SIZE=0x1000
    PICO_USE_STACK_GUARDS=1
    GPT_MATMUL_UNROLL4=$<BOOL:${GPT_MATMUL_UNROLL4}>
)
target_link_options(PicoGPT PRIVATE -Wl,--print-memory-usage)

# Use -DPICO_DEFAULT_BINARY_TYPE=copy_to_ram for SRAM-resident code and weights.
# Without it, the SDK keeps the default flash/XIP layout.
pico_set_program_name(PicoGPT "PicoGPT")
pico_set_program_version(PicoGPT "0.1")
pico_enable_stdio_uart(PicoGPT 0)
pico_enable_stdio_usb(PicoGPT 1)
pico_add_extra_outputs(PicoGPT)

get_filename_component(TOOLCHAIN_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
find_program(PICOGPT_SIZE arm-none-eabi-size HINTS "${TOOLCHAIN_BIN}" REQUIRED)
add_custom_target(memory-report
    COMMAND "${PICOGPT_SIZE}" -A "$<TARGET_FILE:PicoGPT>"
    DEPENDS PicoGPT
    VERBATIM
)
