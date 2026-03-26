cmake_minimum_required(VERSION 3.24)

project(mppi_metal LANGUAGES CXX OBJCXX)

include(CTest)

# Export compile commands for clangd and other tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Metal shader compilation
# ---------------------------------------------------------------------------

set(METAL_STD macos-metal2.3)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Library metallib (mppi_rollout + mppi_reduce kernels)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.air
    COMMAND xcrun metal -std=${METAL_STD} -c
            ${SHADER_DIR}/mppi_rollout.metal
            -I ${SHADER_DIR}
            -o ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.air
    DEPENDS ${SHADER_DIR}/mppi_rollout.metal ${SHADER_DIR}/mppi_core.h ${SHADER_DIR}/philox.h
    COMMENT "Compiling mppi_rollout.metal"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mppi_reduce.air
    COMMAND xcrun metal -std=${METAL_STD} -c
            ${SHADER_DIR}/mppi_reduce.metal
            -I ${SHADER_DIR}
            -o ${CMAKE_CURRENT_BINARY_DIR}/mppi_reduce.air
    DEPENDS ${SHADER_DIR}/mppi_reduce.metal ${SHADER_DIR}/mppi_core.h ${SHADER_DIR}/philox.h
    COMMENT "Compiling mppi_reduce.metal"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.metallib
    COMMAND xcrun metallib
            ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.air
            ${CMAKE_CURRENT_BINARY_DIR}/mppi_reduce.air
            -o ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.metallib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.air
            ${CMAKE_CURRENT_BINARY_DIR}/mppi_reduce.air
    COMMENT "Linking mppi_rollout.metallib"
)
add_custom_target(mppi_rollout_metallib DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.metallib
)

# Test user metallib (test_user_dynamics)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.air
    COMMAND xcrun metal -std=${METAL_STD} -c
            ${SHADER_DIR}/test_user_dynamics.metal
            -I ${SHADER_DIR}
            -o ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.air
    DEPENDS ${SHADER_DIR}/test_user_dynamics.metal ${SHADER_DIR}/mppi_core.h
    COMMENT "Compiling test_user_dynamics.metal"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.metallib
    COMMAND xcrun metallib
            ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.air
            -o ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.metallib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.air
    COMMENT "Linking test_user_dynamics.metallib"
)
add_custom_target(test_user_metallib DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.metallib
)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

add_library(mppi_metal STATIC
    src/mppi_driver.cpp
    src/metal_backend.mm
)

add_dependencies(mppi_metal mppi_rollout_metallib)

target_include_directories(mppi_metal
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(mppi_metal PRIVATE
    MPPI_ROLLOUT_METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/mppi_rollout.metallib"
)

# Enable ARC for Objective-C++ implementation files.
set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "Metal/Foundation frameworks not found. This project requires macOS.")
endif()

target_link_libraries(mppi_metal
    PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(BUILD_TESTING)
    add_executable(mppi_metal_tests
        tests/mppi_driver_tests.cpp
    )

    add_dependencies(mppi_metal_tests test_user_metallib)

    target_link_libraries(mppi_metal_tests
        PRIVATE
            mppi_metal
    )

    target_compile_definitions(mppi_metal_tests PRIVATE
        TEST_USER_METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/test_user_dynamics.metallib"
    )

    add_test(NAME mppi_metal_tests COMMAND mppi_metal_tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
add_subdirectory(examples/nonholonomic_vehicle)
add_subdirectory(examples/nonholonomic_vehicle_concurrent)
add_subdirectory(examples/nonholonomic_vehicle_batch)
add_subdirectory(examples/wez_environment_batch)