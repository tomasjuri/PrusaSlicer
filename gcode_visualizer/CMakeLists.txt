cmake_minimum_required(VERSION 3.16)
project(GCodeVisualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Add NanoSVG dependency (just need the headers)
set(NANOSVG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bundled_deps/nanosvg" CACHE PATH "Path to NanoSVG headers")
if(NOT EXISTS "${NANOSVG_INCLUDE_DIR}")
    # Try alternative path
    set(NANOSVG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/+NanoSVG/nanosvg")
endif()

# We'll download NanoSVG headers directly since they're header-only
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nanosvg")
    message(STATUS "Downloading NanoSVG headers...")
    file(DOWNLOAD "https://raw.githubusercontent.com/fltk/nanosvg/abcd277ea45e9098bed752cf9c6875b533c0892f/src/nanosvg.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/nanosvg/nanosvg.h")
    file(DOWNLOAD "https://raw.githubusercontent.com/fltk/nanosvg/abcd277ea45e9098bed752cf9c6875b533c0892f/src/nanosvgrast.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/nanosvg/nanosvgrast.h")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/libvgcode/glad/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nanosvg)

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
    PrinterConfig.cpp
    ImageExporter.cpp
    TestCubeRenderer.cpp
    GCodePathRenderer.cpp
)

# Create main executable
add_executable(gcode_visualizer 
    ${GCODE_VISUALIZER_SOURCES}
    ${GLAD_SOURCES}
)

# Create printer config example executable
add_executable(example_printer_config
    example_printer_config.cpp
    BedRenderer.cpp
    PrinterConfig.cpp
    ${GLAD_SOURCES}
)

# Add preprocessor definitions
target_compile_definitions(gcode_visualizer PRIVATE
    GL_SILENCE_DEPRECATION
)

target_compile_definitions(example_printer_config PRIVATE
    GL_SILENCE_DEPRECATION
)

# Link libraries
target_link_libraries(gcode_visualizer 
    OpenGL::GL
    glfw
)

target_link_libraries(example_printer_config
    OpenGL::GL
    glfw
)

# Install
install(TARGETS gcode_visualizer example_printer_config DESTINATION bin)

# Print build information
message(STATUS "G-Code Visualizer build configuration:")
message(STATUS "  OpenGL: ${OPENGL_FOUND}")
message(STATUS "  GLFW: ${glfw3_FOUND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Using simple G-code parser") 