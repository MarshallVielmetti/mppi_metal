add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.air
    COMMAND xcrun metal -std=${METAL_STD} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/nonholonomic_vehicle.metal
            -I ${CMAKE_CURRENT_SOURCE_DIR}/../../shaders
            -o ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nonholonomic_vehicle.metal ${CMAKE_CURRENT_SOURCE_DIR}/../../shaders/mppi_core.h
    COMMENT "Compiling nonholonomic_vehicle.metal"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.metallib
    COMMAND xcrun metallib
            ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.air
            -o ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.metallib
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.air
    COMMENT "Linking nonholonomic_vehicle.metallib"
)
add_custom_target(nonholonomic_vehicle_concurrent_metallib DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.metallib
)

add_executable(nonholonomic_vehicle_concurrent_example main.cpp)
add_dependencies(nonholonomic_vehicle_concurrent_example nonholonomic_vehicle_concurrent_metallib)

target_link_libraries(nonholonomic_vehicle_concurrent_example PRIVATE mppi_metal)

# Pass the metallib path as a compile definition so main.cpp can find it
target_compile_definitions(nonholonomic_vehicle_concurrent_example PRIVATE 
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/nonholonomic_vehicle.metallib"
)
