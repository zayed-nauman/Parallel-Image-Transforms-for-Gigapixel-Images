cmake_minimum_required(VERSION 3.16)
project(parallel_image_transforms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # use -std=c++17 not -std=gnu++17

# Release build with optimisations by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT APPLE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# ── Source directory shorthand ────────────────────────────────────────────────
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# libtiff — handle Homebrew prefix on macOS
if(APPLE)
  # Homebrew on Apple Silicon
  set(HOMEBREW_PREFIX "/opt/homebrew" CACHE PATH "Homebrew prefix")
  # Fallback for Intel Macs
  if(NOT EXISTS "${HOMEBREW_PREFIX}/include/tiff.h")
    set(HOMEBREW_PREFIX "/usr/local")
  endif()
  set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

find_path(TIFF_INCLUDE_DIR tiff.h)
find_library(TIFF_LIBRARY  NAMES tiff)
if(NOT TIFF_INCLUDE_DIR OR NOT TIFF_LIBRARY)
  message(FATAL_ERROR
    "libtiff not found.\n"
    "  macOS:  brew install libtiff\n"
    "  Linux:  sudo apt install libtiff-dev")
endif()

# ── Milestone 1 ───────────────────────────────────────────────────────────────
# CPU-only foundation: tiled reader/writer, transform chain, thread pool.
set(M1_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/tile_reader.cpp
    ${SRC_DIR}/tile_writer.cpp
    ${SRC_DIR}/tile_processor.cpp
    ${SRC_DIR}/transforms.cpp
    ${SRC_DIR}/overlap.cpp
)

add_executable(milestone1 ${M1_SOURCES})
target_include_directories(milestone1 PRIVATE ${SRC_DIR} ${TIFF_INCLUDE_DIR})
target_link_libraries(milestone1 PRIVATE Threads::Threads ${TIFF_LIBRARY})

# ── Milestone 2 ───────────────────────────────────────────────────────────────
# Adds: sequential processor, memory manager, work scheduler, GPU tile processor.
set(M2_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/tile_reader.cpp
    ${SRC_DIR}/tile_writer.cpp
    ${SRC_DIR}/tile_processor.cpp
    ${SRC_DIR}/transforms.cpp
    ${SRC_DIR}/overlap.cpp
    ${SRC_DIR}/sequential_processor.cpp
    ${SRC_DIR}/memory_manager.cpp
    ${SRC_DIR}/work_scheduler.cpp
    ${SRC_DIR}/gpu_tile_processor_fixed.cpp
)

add_executable(milestone2 ${M2_SOURCES})
target_include_directories(milestone2 PRIVATE ${SRC_DIR} ${TIFF_INCLUDE_DIR})
target_link_libraries(milestone2 PRIVATE Threads::Threads ${TIFF_LIBRARY})

# ── Milestone 3 ───────────────────────────────────────────────────────────────
# Adds optional performance/pipeline files on top of milestone 2.
# Files are optional so older submissions still build if a file is temporarily
# omitted during development.
set(M3_SOURCES ${M2_SOURCES})

foreach(optional_m3_source
    pipeline_fusion.cpp
    performance_metrics.cpp
    simd_kernels.cpp
    pipeline_report.cpp
    benchmark_sweep.cpp
)
  if(EXISTS "${SRC_DIR}/${optional_m3_source}")
    list(APPEND M3_SOURCES "${SRC_DIR}/${optional_m3_source}")
  else()
    message(STATUS "Milestone 3: optional source not found, skipping: ${optional_m3_source}")
  endif()
endforeach()

add_executable(milestone3 ${M3_SOURCES})
target_include_directories(milestone3 PRIVATE ${SRC_DIR} ${TIFF_INCLUDE_DIR})
target_link_libraries(milestone3 PRIVATE Threads::Threads ${TIFF_LIBRARY})

# ── Test image generator (Python helper, not compiled) ────────────────────────
# Run:  python3 tools/gen_test_image.py <width> <height>
