#pragma once

#include <memory>
#include <string>
#include <array>

// OpenGL and GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

// Our simple G-code parser
#include "SimpleGCodeParser.hpp"

// Our components
#include "CameraController.hpp"
#include "ImageExporter.hpp"
#include "BedRenderer.hpp"
#include "TestCubeRenderer.hpp"
#include "GCodePathRenderer.hpp"

// Simple matrix type for our camera
struct Mat4x4 {
    float m_data[16];
    
    Mat4x4() {
        // Initialize as identity matrix
        for (int i = 0; i < 16; i++) m_data[i] = 0.0f;
        m_data[0] = m_data[5] = m_data[10] = m_data[15] = 1.0f;
    }
    
    const float* ptr() const { return m_data; }
    const float* data() const { return m_data; }  // Compatibility with existing API
};

class GCodeVisualizerApp {
public:
    GCodeVisualizerApp();
    ~GCodeVisualizerApp();
    
    // Initialize the application (OpenGL context, etc.)
    bool initialize();
    
    // Load a G-code file
    bool loadGCode(const std::string& filename);
    
    // Render and save the visualization
    bool renderAndSave();
    
    // Cleanup resources
    void cleanup();
    
private:
    // Constants
    static constexpr int WINDOW_WIDTH = 1920;
    static constexpr int WINDOW_HEIGHT = 1080;
    
    // GLFW and OpenGL
    GLFWwindow* m_window = nullptr;
    bool m_gl_initialized = false;
    
    // Offscreen rendering
    GLuint m_framebuffer = 0;
    GLuint m_color_texture = 0;
    GLuint m_depth_renderbuffer = 0;
    
    // G-code parser
    std::unique_ptr<SimpleGCode::GCodeParser> m_gcode_parser;
    
    // Our components
    std::unique_ptr<CameraController> m_camera_controller;
    std::unique_ptr<ImageExporter> m_image_exporter;
    std::unique_ptr<BedRenderer> m_bed_renderer;
    std::unique_ptr<TestCubeRenderer> m_test_cube_renderer;
    std::unique_ptr<GCodePathRenderer> m_gcode_path_renderer;
    
    // State
    bool m_gcode_loaded = false;
    
    // Helper methods
    bool initializeGLFW();
    bool initializeOpenGL();
    bool createOffscreenFramebuffer();
    void cleanupOffscreenFramebuffer();
    void setupTopViewCamera();
    void renderScene();
}; 