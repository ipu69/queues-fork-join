cmake_minimum_required(VERSION 3.18)
project(fjqsim VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

include_directories(include)

add_subdirectory(fjqsim)

# -------------------------------------------
# All the following is used in C++ project 
# solely, not in Python bindings:
#
# - main() for simulation executable
# - tests
# - benchmarks
# ------------------------------------------

add_executable(fjqsim main.cpp)
target_link_libraries(fjqsim fjqsimlib)

# We recommend to install google test framework
# in $HOME/.local, while it isn't mandatory.
include_directories(include)
include_directories($ENV{HOME}/.local/include)
link_directories($ENV{HOME}/.local/lib)

# Build tests if google test framework is installed
find_package(GTest)
if(GTEST_FOUND)
	add_subdirectory(tests)
endif()
