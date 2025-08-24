cmake_minimum_required(VERSION 3.18...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_GENERATOR_PLATFORM x64)

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

# - Avoid issues with CMAKE_GENERATOR_PLATFORM from cibuildwheels
if(NOT WIN32)
  unset(CMAKE_GENERATOR_PLATFORM)
endif()

# - Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Creating a ${CMAKE_BUILD_TYPE} build")

# - Find packages
find_package(OpenMP REQUIRED CXX)
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# - Find module source-file
# The build command `pip install ...` re-runs cmake on every call!
# Treats all cpp files as extension modules.
# Includes all top-level .h files as source file.
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME})
file(GLOB_RECURSE EXT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${SRC_PATH}/*.cpp)
foreach(EXT_SOURCE ${EXT_SOURCES})
  # Create variables for the extension name, stub, and destination
  get_filename_component(EXT_FILE ${EXT_SOURCE} NAME_WE)
  get_filename_component(EXT_DIR ${EXT_SOURCE} DIRECTORY)
  cmake_path(RELATIVE_PATH EXT_DIR BASE_DIRECTORY ${SRC_PATH})
  set(EXT_NAME ${EXT_FILE}_ext)
  set(EXT_STUB ${EXT_FILE}_stub)
  set(EXT_STUB_OUT ${EXT_NAME}.pyi)
  set(EXT_DOCS_OUT ${EXT_FILE}_docs.py)
  set(EXT_MOD_OUT "${EXT_FILE}.py")
  set(EXT_DESTINATION ${SKBUILD_PROJECT_NAME}/${EXT_DIR})
  message(STATUS "Found extension: ${EXT_DESTINATION}${EXT_NAME}")

  # Add speed optimized module target
  nanobind_add_module(${EXT_NAME}
    NOMINSIZE
    STABLE_ABI
    FREE_THREADED
    LTO
    NB_DOMAIN ${SKBUILD_PROJECT_NAME}
    ${EXT_SOURCE}
  )
  target_link_libraries(${EXT_NAME} PRIVATE OpenMP::OpenMP_CXX)
  nanobind_add_stub(
    ${EXT_STUB}
    MODULE ${EXT_NAME}
    OUTPUT ${EXT_STUB_OUT}
    PYTHON_PATH $<TARGET_FILE_DIR:${EXT_NAME}>
    DEPENDS ${EXT_NAME}
  )
  # Runs postprocessing on the generated stub file!
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_DOCS_OUT}
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_MOD_OUT}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrap_stubs.py
      ${CMAKE_CURRENT_BINARY_DIR}
      ${EXT_STUB_OUT}
      ${EXT_NAME}
      ${EXT_DOCS_OUT}
      ${EXT_MOD_OUT}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_STUB_OUT}
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrap_stubs.py
    VERBATIM
  )
  add_custom_target(
    ${EXT_NAME}_docs ALL
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_DOCS_OUT}
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_MOD_OUT}
  )
  install(TARGETS ${EXT_NAME} LIBRARY DESTINATION ${EXT_DESTINATION})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${EXT_DOCS_OUT} DESTINATION ${EXT_DESTINATION})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${EXT_MOD_OUT} DESTINATION ${EXT_DESTINATION})
endforeach()
