cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2_w CACHE STRING "Board type")

if(DEFINED ENV{PICO_SDK_PATH} AND NOT PICO_SDK_PATH)
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif()
if(NOT PICO_SDK_PATH)
    message(FATAL_ERROR "Set PICO_SDK_PATH to a pico-sdk (>= 2.0) checkout")
endif()
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(streampilot_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

add_executable(streampilot_pico src/main.c src/protocol.c src/stream_ac.c)
target_compile_options(streampilot_pico PRIVATE -O2 -Wall -Wextra)
set(SAC_BAUD 921600 CACHE STRING "UART baud rate")
target_compile_definitions(streampilot_pico PRIVATE SAC_BAUD=${SAC_BAUD})
target_link_libraries(streampilot_pico pico_stdlib hardware_uart m)

# UART0 carries the protocol, so no stdio on it (or on USB).
pico_enable_stdio_uart(streampilot_pico 0)
pico_enable_stdio_usb(streampilot_pico 0)
pico_add_extra_outputs(streampilot_pico)
