#pragma once

#include <glad/gl.h>
#include <vector>

// Forward declare our simple matrix type
struct Mat4x4;

// Simple bed renderer based on PrusaSlicer's 3D bed implementation
class BedRenderer {
public:
    BedRenderer();
    ~BedRenderer();
    
    // Initialize the bed with given dimensions
    void initialize(float width = 250.0f, float height = 210.0f, float grid_spacing = 10.0f);
    
    // Render the bed grid and surface
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // Cleanup OpenGL resources
    void cleanup();
    
private:
    // OpenGL objects
    GLuint m_grid_vao = 0;
    GLuint m_grid_vbo = 0;
    GLuint m_surface_vao = 0;
    GLuint m_surface_vbo = 0;
    GLuint m_surface_ebo = 0;
    
    // Shader program for rendering
    GLuint m_shader_program = 0;
    
    // Geometry data
    std::vector<float> m_grid_vertices;
    std::vector<float> m_surface_vertices;
    std::vector<unsigned int> m_surface_indices;
    
    int m_grid_vertex_count = 0;
    int m_surface_index_count = 0;
    
    bool m_initialized = false;
    
    // Bed dimensions
    float m_width = 250.0f;
    float m_height = 210.0f;
    float m_grid_spacing = 10.0f;
    
    // Helper methods
    void createGridGeometry();
    void createSurfaceGeometry();
    void createShaders();
    void setupGridBuffers();
    void setupSurfaceBuffers();
    
    // Shader source code
    static const char* getVertexShaderSource();
    static const char* getFragmentShaderSource();
    
    // Shader utilities
    GLuint compileShader(GLenum type, const char* source);
    GLuint createShaderProgram(const char* vertex_source, const char* fragment_source);
}; 