#pragma once

#include <vector>
#include <memory>
#include "SimpleGCodeParser.hpp"

// Include GLAD before any OpenGL headers to avoid conflicts  
#define GLFW_INCLUDE_NONE
#include <glad/gl.h>

// Forward declare our simple matrix type
struct Mat4x4;

class GCodePathRenderer {
public:
    GCodePathRenderer();
    ~GCodePathRenderer();
    
    // Initialize OpenGL resources
    bool initialize();
    
    // Load G-code moves for rendering
    void setGCodeMoves(const std::vector<SimpleGCode::GCodeMove>& moves);
    
    // Render all G-code paths
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // Cleanup OpenGL resources
    void cleanup();
    
private:
    struct PathVertex {
        float x, y, z;
        float r, g, b;  // Color based on move type
    };
    
    // OpenGL objects
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_shader_program = 0;
    
    // Vertex data
    std::vector<PathVertex> m_vertices;
    size_t m_vertex_count = 0;
    
    // Shader creation
    bool createShaders();
    GLuint compileShader(const char* source, GLenum type);
    
    // Convert moves to renderable lines
    void generatePathGeometry(const std::vector<SimpleGCode::GCodeMove>& moves);
    
    // Get color based on move type
    void getMoveColor(SimpleGCode::MoveType type, float& r, float& g, float& b);
}; 