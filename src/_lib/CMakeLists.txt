# - Find module source-file
# The build command `pip install ...` re-runs cmake on every call!
# Treats all cpp files as extension modules.
# Includes all top-level .h files as source file.
file(GLOB_RECURSE EXT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(EXT_SOURCE ${EXT_SOURCES})
  # Create variables for the extension name, stub, and destination
  get_filename_component(EXT_FILE ${EXT_SOURCE} NAME_WE)
  get_filename_component(EXT_DIR ${EXT_SOURCE} DIRECTORY)
  cmake_path(RELATIVE_PATH EXT_DIR BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/wrap_stubs.py
      ${CMAKE_CURRENT_BINARY_DIR}
      ${EXT_STUB_OUT}
      ${EXT_NAME}
      ${EXT_DOCS_OUT}
      ${EXT_MOD_OUT}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/${EXT_STUB_OUT}
      ${CMAKE_SOURCE_DIR}/cmake/wrap_stubs.py
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
