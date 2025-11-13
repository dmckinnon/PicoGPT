cmake_minimum_required(VERSION 3.12)

# Project configuration
project(PicoGPT C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compiler flags for debugging and warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Create stub headers directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/stubs)

# Create stub header files for Pico SDK
file(WRITE ${CMAKE_BINARY_DIR}/stubs/pico/stdlib.h "
#ifndef PICO_STDLIB_H
#define PICO_STDLIB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>

// Stub implementations for Linux
static inline void stdio_init_all(void) {
    // No-op for Linux
}

static inline void sleep_ms(uint32_t ms) {
    usleep(ms * 1000);
}

typedef uint32_t alarm_id_t;
typedef int64_t (*alarm_callback_t)(alarm_id_t id, void *user_data);

static inline alarm_id_t add_alarm_in_ms(uint32_t ms, alarm_callback_t callback, void *user_data, bool fire_if_past) {
    // Stub implementation - just return 0
    (void)ms; (void)callback; (void)user_data; (void)fire_if_past;
    return 0;
}

static inline uint64_t time_us_64(void) {
    // Return time in microseconds using clock()
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

#endif
")

file(WRITE ${CMAKE_BINARY_DIR}/stubs/hardware/timer.h "
#ifndef HARDWARE_TIMER_H
#define HARDWARE_TIMER_H

#include <stdint.h>
#include <stdbool.h>

// Stub definitions for timer functionality
typedef uint32_t alarm_id_t;
typedef int64_t (*alarm_callback_t)(alarm_id_t id, void *user_data);

#endif
")

file(WRITE ${CMAKE_BINARY_DIR}/stubs/hardware/clocks.h "
#ifndef HARDWARE_CLOCKS_H
#define HARDWARE_CLOCKS_H

#include <stdint.h>

// Stub definitions for clock functionality
typedef enum {
    clk_sys,
    clk_usb
} clock_index_t;

static inline uint32_t clock_get_hz(clock_index_t clock) {
    // Return dummy values for Linux testing
    switch(clock) {
        case clk_sys: return 125000000; // 125 MHz
        case clk_usb: return 48000000;  // 48 MHz
        default: return 0;
    }
}

#endif
")

# Add include directories
include_directories(${CMAKE_BINARY_DIR}/stubs)
include_directories(${CMAKE_SOURCE_DIR})

# Add executable
add_executable(PicoGPT
    PicoGPT.c
    attention_test.c
)

# Link math library
target_link_libraries(PicoGPT m)

# Set output directory
set_target_properties(PicoGPT PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)