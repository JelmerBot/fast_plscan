cmake_minimum_required(VERSION 3.18...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# - Warn if the user invokes CMake directly
if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add --config-settings editable.rebuild=true to 
  auto-rebuild when the package is imported. Otherwise, you need to 
  rerun the above after editing C++ files.")
endif()

# - Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Creating a ${CMAKE_BUILD_TYPE} build")

# - Options
set(PLSCAN_CREATE_STUBS ON CACHE BOOL "Create nanobind stub files for IDE support")
set(PLSCAN_COVERAGE OFF CACHE BOOL "Enable C++ coverage instrumentation")

# - Find packages
find_package(OpenMP REQUIRED CXX)
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# - Configure compiler
add_library(compiler_options INTERFACE)
target_compile_features(compiler_options INTERFACE cxx_std_23)

if(PLSCAN_COVERAGE)
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    message(WARNING "PLSCAN_COVERAGE is ON but CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
      "Coverage data will be inaccurate due to compiler optimizations. "
      "Set -DCMAKE_BUILD_TYPE=Debug for reliable results.")
  endif()
  if(WIN32)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      message(FATAL_ERROR "PLSCAN_COVERAGE requires clang-cl on Windows. "
        "Reconfigure with -T ClangCL")
    endif()
    target_compile_options(compiler_options INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(compiler_options INTERFACE -fprofile-instr-generate)
  else()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(compiler_options INTERFACE --coverage)
      target_link_options(compiler_options INTERFACE --coverage)
    else()
      message(WARNING "PLSCAN_COVERAGE: unsupported compiler ${CMAKE_CXX_COMPILER_ID}, no flags added.")
    endif()
  endif()
endif()

# - Add nanobind modules
add_subdirectory(src/${SKBUILD_PROJECT_NAME}/_api)