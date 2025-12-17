# - Find module source-files
# The build command `pip install ...` re-runs cmake on every call!
# Combines all .cpp files into a single nanobind module.
get_filename_component(EXT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
file(GLOB_RECURSE EXT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.cpp)
message(STATUS "Found extension ${EXT_NAME}: ${EXT_SOURCES}")

nanobind_add_module(${EXT_NAME}
        NOMINSIZE
        STABLE_ABI
        FREE_THREADED
        LTO
        NB_DOMAIN ${SKBUILD_PROJECT_NAME}
        ${EXT_SOURCES}
)
target_link_libraries(${EXT_NAME} PRIVATE compiler_options OpenMP::OpenMP_CXX)

nanobind_add_stub(
        ${EXT_NAME}_stub
        MODULE ${EXT_NAME}
        OUTPUT ${EXT_NAME}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${EXT_NAME}>
        DEPENDS ${EXT_NAME}
)
install(TARGETS ${EXT_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${EXT_NAME}.pyi DESTINATION ${SKBUILD_PROJECT_NAME})