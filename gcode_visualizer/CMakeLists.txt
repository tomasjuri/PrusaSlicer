cmake_minimum_required(VERSION 3.16)
project(GCodeVisualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/libvgcode/glad/include)

# Add GLAD OpenGL loader
set(GLAD_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/libvgcode/glad/src/gl.c
)

# Source files for our G-code visualizer
set(GCODE_VISUALIZER_SOURCES
    main.cpp
    GCodeVisualizerApp.cpp
    SimpleGCodeParser.cpp
    CameraController.cpp
    BedRenderer.cpp
    ImageExporter.cpp
    TestCubeRenderer.cpp
)

# Create executable
add_executable(gcode_visualizer 
    ${GCODE_VISUALIZER_SOURCES}
    ${GLAD_SOURCES}
)

# Add preprocessor definitions
target_compile_definitions(gcode_visualizer PRIVATE
    GL_SILENCE_DEPRECATION
)

# Link libraries
target_link_libraries(gcode_visualizer 
    OpenGL::GL
    glfw
)

# Install
install(TARGETS gcode_visualizer DESTINATION bin)

# Print build information
message(STATUS "G-Code Visualizer build configuration:")
message(STATUS "  OpenGL: ${OPENGL_FOUND}")
message(STATUS "  GLFW: ${glfw3_FOUND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Using simple G-code parser") 