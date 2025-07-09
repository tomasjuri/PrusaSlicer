#pragma once

#include <glad/gl.h>
#include <vector>
#include <string>
#include "PrinterConfig.hpp"

// Forward declare our simple matrix type
struct Mat4x4;

// Professional bed renderer that mimics PrusaSlicer's approach
class BedRenderer {
public:
    BedRenderer();
    ~BedRenderer();
    
    // Initialize with professional bed rendering system
    bool initialize(const std::string& stl_path = "", const std::string& svg_path = "");
    
    // Initialize using printer model configuration (new method!)
    bool initializeFromPrinterModel(const std::string& printer_model, const std::string& config_file_path = "");
    
    // Set bed shape for dynamic geometry
    void setBedShape(const std::vector<std::pair<float, float>>& bed_shape);
    
    // Render the bed using professional approach
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // Cleanup OpenGL resources
    void cleanup();
    
private:
    // OpenGL objects for STL model
    GLuint m_model_vao = 0;
    GLuint m_model_vbo = 0;
    GLuint m_model_ebo = 0;
    
    // OpenGL objects for bed triangles (textured surface)
    GLuint m_triangle_vao = 0;
    GLuint m_triangle_vbo = 0;
    GLuint m_triangle_ebo = 0;
    
    // OpenGL objects for grid overlay
    GLuint m_grid_vao = 0;
    GLuint m_grid_vbo = 0;
    
    // Texture object
    GLuint m_texture_id = 0;
    
    // Professional shader programs (matching main slicer)
    GLuint m_model_shader = 0;      // Gouraud lighting for STL models
    GLuint m_texture_shader = 0;    // Printbed shader for textured surface
    GLuint m_grid_shader = 0;       // Flat shader for grid lines
    
    // Model geometry data
    std::vector<float> m_model_vertices;
    std::vector<unsigned int> m_model_indices;
    
    // Bed triangle geometry (textured surface)
    std::vector<float> m_triangle_vertices;
    std::vector<unsigned int> m_triangle_indices;
    
    // Grid geometry
    std::vector<float> m_grid_vertices;
    
    int m_model_vertex_count = 0;
    int m_model_index_count = 0;
    int m_triangle_vertex_count = 0;
    int m_triangle_index_count = 0;
    int m_grid_vertex_count = 0;
    
    // Model offset (like main slicer)
    float m_model_offset_x = 0.0f;
    float m_model_offset_y = 0.0f;
    float m_model_offset_z = 0.0f;
    
    // SVG texture dimensions in millimeters (for proper scaling)
    float m_svg_width_mm = 0.0f;
    float m_svg_height_mm = 0.0f;
    
    bool m_initialized = false;
    bool m_has_model = false;
    
    // File paths
    std::string m_stl_path;
    std::string m_svg_path;
    
    // Configuration system
    PrinterConfig m_printer_config;
    
    // Bed shape points (dynamic)
    std::vector<std::pair<float, float>> m_bed_shape;
    
    // Rendering methods (split like main slicer)
    void renderModel(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    void renderTexturedModel(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);  // Fixed method!
    void renderTexture(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    void renderGrid(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    void renderTextureWithGrid(const Mat4x4& view_matrix, const Mat4x4& projection_matrix);
    
    // STL loading
    bool loadSTLModel(const std::string& stl_path);
    
    // Procedural geometry creation (like main slicer)
    void createBedTriangles();
    void createProfessionalGrid();
    void createProfessionalTexture();
    
    // Shader creation
    bool createShaders();
    void setupModelBuffers();
    void setupTriangleBuffers();
    void setupGridBuffers();
    
    // Professional shader source code (matching main slicer)
    static const char* getPrintbedVertexShader();
    static const char* getPrintbedFragmentShader();
    static const char* getGouraudVertexShader();
    static const char* getGouraudFragmentShader();
    static const char* getFlatVertexShader();
    static const char* getFlatFragmentShader();
    
    // Texture utilities  
    bool createProceduralTexture();
    bool loadSVGTexture(const std::string& svg_path);
    bool loadImageTexture(const std::string& image_path);
    GLuint loadTextureFromData(const unsigned char* data, int width, int height, int channels);
    
    // Real SVG parsing using NanoSVG (like PrusaSlicer)
    // Removed fake texture methods - now using actual SVG parsing!
    
    // Shader utilities
    GLuint compileShader(GLenum type, const char* source);
    GLuint createShaderProgram(const char* vertex_source, const char* fragment_source);
    
    // Polygon triangulation
    void triangulatePolygon(const std::vector<std::pair<float, float>>& shape, 
                           float min_x, float min_y, float width, float height);
}; 