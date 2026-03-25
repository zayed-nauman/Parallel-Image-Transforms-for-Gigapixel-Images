cmake_minimum_required(VERSION 3.16)
project(milestone1 CXX)

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

# ── Dependencies ─────────────────────────────────────────────────────────────
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

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/tile_reader.cpp
    src/tile_writer.cpp
    src/tile_processor.cpp
    src/transforms.cpp
    src/overlap.cpp
)

add_executable(milestone1 ${SOURCES})

target_include_directories(milestone1 PRIVATE src ${TIFF_INCLUDE_DIR})
target_link_libraries(milestone1 PRIVATE Threads::Threads ${TIFF_LIBRARY})

# ── Test image generator (Python helper, not compiled) ────────────────────────
# Run:  python3 tools/gen_test_image.py