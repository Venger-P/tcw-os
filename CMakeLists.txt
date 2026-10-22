cmake_minimum_required(VERSION 3.28.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tcw_demo)

# TFLite Micro requires C++ without thread-safe statics
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

target_sources(app PRIVATE
    src/main.c
    src/tcw_runtime.c
    src/tcw_ai.cpp
    src/model.cpp
    src/constants.c
    src/assert.cpp
)

target_include_directories(app PRIVATE src)
