#pragma once

#include <glad/gl.h>
#include <vector>
#include <string>

// Forward declare our simple matrix type
struct Mat4x4;

// Advanced bed renderer that can load STL models and SVG textures like PrusaSlicer
class BedRenderer {
public:
    BedRenderer();
    ~BedRenderer();
    
    // Initialize with Prusa bed assets
    bool initialize(const std::string& stl_path = "", const std::string& svg_path = "");
    
    // Render the bed model and texture
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // Cleanup OpenGL resources
    void cleanup();
    
private:
    // OpenGL objects for STL model
    GLuint m_model_vao = 0;
    GLuint m_model_vbo = 0;
    GLuint m_model_ebo = 0;
    
    // OpenGL objects for grid overlay
    GLuint m_grid_vao = 0;
    GLuint m_grid_vbo = 0;
    
    // Texture objects
    GLuint m_texture_id = 0;
    
    // Shader programs
    GLuint m_model_shader = 0;
    GLuint m_grid_shader = 0;
    
    // Model geometry data
    std::vector<float> m_model_vertices;
    std::vector<unsigned int> m_model_indices;
    std::vector<float> m_grid_vertices;
    
    int m_model_vertex_count = 0;
    int m_model_index_count = 0;
    int m_grid_vertex_count = 0;
    
    bool m_initialized = false;
    bool m_has_model = false;
    bool m_has_texture = false;
    
    // File paths
    std::string m_stl_path;
    std::string m_svg_path;
    
    // STL loading
    bool loadSTLModel(const std::string& stl_path);
    void parseSTLTriangle(const std::string& line, std::vector<float>& vertices);
    
    // SVG texture loading (convert to PNG first)
    bool loadSVGTexture(const std::string& svg_path);
    bool loadPNGTexture(const std::string& png_path);
    
    // Grid generation for overlay
    void createGridOverlay();
    
    // Shader creation
    bool createShaders();
    void setupModelBuffers();
    void setupGridBuffers();
    
    // Shader source code
    static const char* getModelVertexShader();
    static const char* getModelFragmentShader();
    static const char* getGridVertexShader();
    static const char* getGridFragmentShader();
    
    // Shader utilities
    GLuint compileShader(GLenum type, const char* source);
    GLuint createShaderProgram(const char* vertex_source, const char* fragment_source);
}; 