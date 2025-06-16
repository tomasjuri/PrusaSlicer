#pragma once

#include <glad/gl.h>
#include <vector>

// Forward declare our simple matrix type
struct Mat4x4;

// Simple test cube renderer to verify the rendering pipeline works
class TestCubeRenderer {
public:
    TestCubeRenderer();
    ~TestCubeRenderer();
    
    // Initialize the cube geometry and shaders
    void initialize();
    
    // Render a red cube at the center of the scene
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // Cleanup OpenGL resources
    void cleanup();
    
private:
    // OpenGL objects
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_ebo = 0;
    GLuint m_shader_program = 0;
    
    // Cube geometry
    std::vector<float> m_vertices;
    std::vector<unsigned int> m_indices;
    
    bool m_initialized = false;
    
    // Helper methods
    void createCubeGeometry();
    void createShaders();
    void setupBuffers();
    
    const char* getVertexShaderSource();
    const char* getFragmentShaderSource();
    GLuint compileShader(GLenum type, const char* source);
    GLuint createShaderProgram(const char* vertex_source, const char* fragment_source);
}; 