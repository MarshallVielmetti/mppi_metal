add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.air
    COMMAND xcrun metal -std=${METAL_STD} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/wez_environment.metal
            -I ${CMAKE_CURRENT_SOURCE_DIR}/../../shaders
            -o ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/wez_environment.metal ${CMAKE_CURRENT_SOURCE_DIR}/../../shaders/mppi_core.h
    COMMENT "Compiling wez_environment.metal (batch example)"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.metallib
    COMMAND xcrun metallib
            ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.air
            -o ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.metallib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.air
    COMMENT "Linking wez_environment.metallib (batch example)"
)
add_custom_target(wez_environment_batch_metallib DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/wez_environment.metallib
)

add_executable(wez_environment_batch_example main.cpp)
add_dependencies(wez_environment_batch_example wez_environment_batch_metallib)

target_link_libraries(wez_environment_batch_example PRIVATE mppi_metal)

# Pass the metallib path as a compile definition so main.cpp can find it
target_compile_definitions(wez_environment_batch_example PRIVATE 
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/wez_environment.metallib"
)
