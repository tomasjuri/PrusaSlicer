#include "BedRenderer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <filesystem>

// NanoSVG for real SVG parsing (like PrusaSlicer)
#define NANOSVG_IMPLEMENTATION
#include <nanosvg.h>
#define NANOSVGRAST_IMPLEMENTATION  
#include <nanosvgrast.h>

// Simple matrix type (copied to avoid include conflicts)
struct Mat4x4 {
    float m_data[16];
    
    Mat4x4() {
        // Initialize as identity matrix
        for (int i = 0; i < 16; i++) m_data[i] = 0.0f;
        m_data[0] = m_data[5] = m_data[10] = m_data[15] = 1.0f;
    }
    
    const float* ptr() const { return m_data; }
    const float* data() const { return m_data; }
};

BedRenderer::BedRenderer() {
}

BedRenderer::~BedRenderer() {
    cleanup();
}

bool BedRenderer::initialize(const std::string& stl_path, const std::string& svg_path) {
    std::cout << "Initializing PrusaSlicer-style bed renderer..." << std::endl;
    
    m_stl_path = stl_path;
    m_svg_path = svg_path;
    
    // Create professional shaders matching main slicer
    if (!createShaders()) {
        std::cerr << "Failed to create professional bed shaders" << std::endl;
        return false;
    }
    
    // Load STL model if provided
    if (!stl_path.empty()) {
        std::cout << "Loading bed model: " << stl_path << std::endl;
        if (loadSTLModel(stl_path)) {
            m_has_model = true;
            setupModelBuffers();
            std::cout << "Loaded bed model with " << m_model_vertices.size()/8 << " vertices" << std::endl;
        } else {
            std::cerr << "Failed to load STL model, using procedural bed" << std::endl;
        }
    }
    
    // Create procedural bed triangles (like main slicer)
    createBedTriangles();
    setupTriangleBuffers();
    
    // Create professional grid system
    createProfessionalGrid();
    setupGridBuffers();
    
    // Create professional texture (try SVG first, then fallback)
    if (!svg_path.empty()) {
        if (!loadSVGTexture(svg_path)) {
            std::cout << "Failed to load SVG texture, using procedural fallback" << std::endl;
            createProfessionalTexture();
        }
    } else {
        createProfessionalTexture();
    }
    
    m_initialized = true;
    std::cout << "Professional bed renderer initialized successfully" << std::endl;
    
    return true;
}

bool BedRenderer::initializeFromPrinterModel(const std::string& printer_model, const std::string& config_file_path) {
    std::cout << "Initializing bed renderer for printer model: " << printer_model << std::endl;
    
    // Determine config file path
    std::string config_path = config_file_path;
    if (config_path.empty()) {
        // Try to find the default PrusaResearch config
        config_path = "../resources/profiles/PrusaResearch.ini";
        
        // Also try relative to current directory
        if (!std::ifstream(config_path).good()) {
            config_path = "resources/profiles/PrusaResearch.ini";
        }
        
        // Try relative to project root
        if (!std::ifstream(config_path).good()) {
            config_path = "../../resources/profiles/PrusaResearch.ini";
        }
    }
    
    // Load printer configuration
    if (!m_printer_config.loadConfig(config_path)) {
        std::cerr << "Failed to load printer configuration from: " << config_path << std::endl;
        std::cerr << "Falling back to default bed rendering..." << std::endl;
        return initialize(); // Fallback to default initialization
    }
    
    // Get bed model and texture paths
    std::string stl_path = m_printer_config.getBedModelPath(printer_model);
    std::string svg_path = m_printer_config.getBedTexturePath(printer_model);
    
    if (stl_path.empty() && svg_path.empty()) {
        std::cerr << "No bed model or texture found for printer: " << printer_model << std::endl;
        std::cerr << "Available models: ";
        auto available = m_printer_config.getAvailableModels();
        for (const auto& model : available) {
            std::cerr << model << " ";
        }
        std::cerr << std::endl;
        std::cerr << "Falling back to default bed rendering..." << std::endl;
        return initialize(); // Fallback to default initialization
    }
    
    std::cout << "Found configuration for " << printer_model << ":" << std::endl;
    if (!stl_path.empty()) {
        std::cout << "  STL Model: " << stl_path << std::endl;
    }
    if (!svg_path.empty()) {
        std::cout << "  SVG Texture: " << svg_path << std::endl;
    }
    
    // Get bed shape if available
    auto bed_shape = m_printer_config.getBedShape(printer_model);
    if (!bed_shape.empty()) {
        setBedShape(bed_shape);
    }
    
    // Initialize using the found paths
    return initialize(stl_path, svg_path);
}

void BedRenderer::render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    if (!m_initialized) return;
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // FIXED: Implement PrusaSlicer's proper two-layer system!
    // 1. Black STL model base (solid dark material)
    // 2. Separate flat textured surface on top (with proper SVG scaling)
    
    if (m_has_model && m_model_index_count > 0) {
        // System bed (like PrusaSlicer's render_system):
        // Layer 1: Render STL model as solid black base (like render_model)
        renderModel(view_matrix, projection_matrix);
        
        // Layer 2: Render separate flat textured surface on top (like render_texture)
        renderTexture(view_matrix, projection_matrix);
    } else {
        // Custom bed: render textured surface + grid (like main slicer's render_default)
        renderTextureWithGrid(view_matrix, projection_matrix);
    }
    
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
}

void BedRenderer::renderModel(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    // Render 3D STL bed model as solid black base (like PrusaSlicer's render_model)
    if (!m_has_model || m_model_index_count == 0) return;
    
    glUseProgram(m_model_shader);
    
    // Create view-model matrix with proper offset (like PrusaSlicer)
    Mat4x4 view_model_matrix;
    for (int i = 0; i < 16; i++) {
        view_model_matrix.m_data[i] = view_matrix.data()[i];
    }
    
    // Apply PrusaSlicer's model offset: center + slightly down to avoid Z-fighting
    view_model_matrix.m_data[12] += m_model_offset_x;   // center X
    view_model_matrix.m_data[13] += m_model_offset_y;   // center Y  
    view_model_matrix.m_data[14] += m_model_offset_z;   // move down (-0.03 like PrusaSlicer)
    
    // Set uniforms (only those that exist in the gouraud shader)
    GLint view_loc = glGetUniformLocation(m_model_shader, "view_model_matrix");
    GLint proj_loc = glGetUniformLocation(m_model_shader, "projection_matrix");
    GLint color_loc = glGetUniformLocation(m_model_shader, "uniform_color");
    
    // Set the uniforms
    if (view_loc >= 0) glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_model_matrix.data());
    if (proj_loc >= 0) glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Set model color to be visible (dark gray instead of black)
    if (color_loc >= 0) {
        glUniform4f(color_loc, 0.3f, 0.3f, 0.3f, 1.0f);
    }
    
    // Render STL model as solid dark material
    glBindVertexArray(m_model_vao);
    glDrawElements(GL_TRIANGLES, m_model_index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    glUseProgram(0);
}

void BedRenderer::renderTexturedModel(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    // Render 3D STL bed model with real SVG texture applied directly!
    glUseProgram(m_texture_shader);
    
    // Create proper view-model matrix with STL model offset
    Mat4x4 view_model_matrix;
    for (int i = 0; i < 16; i++) {
        view_model_matrix.m_data[i] = view_matrix.data()[i];
    }
    // Apply model translation offsets to center the bed
    view_model_matrix.m_data[12] += m_model_offset_x;
    view_model_matrix.m_data[13] += m_model_offset_y;
    view_model_matrix.m_data[14] += m_model_offset_z;
    
    // Set uniforms with proper model positioning
    GLint view_loc = glGetUniformLocation(m_texture_shader, "view_model_matrix");
    GLint proj_loc = glGetUniformLocation(m_texture_shader, "projection_matrix");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_model_matrix.data());
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Set texture uniforms for real SVG rendering
    GLint transparent_loc = glGetUniformLocation(m_texture_shader, "transparent_background");
    GLint svg_loc = glGetUniformLocation(m_texture_shader, "svg_source");
    GLint tex_loc = glGetUniformLocation(m_texture_shader, "in_texture");
    
    // Check if uniforms are valid and set them
    if (transparent_loc != -1) glUniform1i(transparent_loc, 0);  // Opaque background
    if (svg_loc != -1) glUniform1i(svg_loc, 1);  // Use SVG-style rendering (we have real SVG!)
    
    // Bind the real SVG texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    if (tex_loc != -1) glUniform1i(tex_loc, 0);
    
    // Render the actual 3D STL bed model with corrected texture coordinates
    if (m_model_vao != 0 && m_model_index_count > 0) {
        glBindVertexArray(m_model_vao);
        glDrawElements(GL_TRIANGLES, m_model_index_count, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
    glUseProgram(0);
}

void BedRenderer::renderTexture(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    // Render flat textured surface on top of black STL base (like PrusaSlicer's render_texture)
    if (!m_has_model || !m_texture_id) return;
    
    glUseProgram(m_texture_shader);
    
    // Use identity matrix for flat surface at Z=0 (above STL base at Z=-0.03)
    GLint view_loc = glGetUniformLocation(m_texture_shader, "view_model_matrix");
    GLint proj_loc = glGetUniformLocation(m_texture_shader, "projection_matrix");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Set texture uniforms (matching PrusaSlicer's printbed shader)
    GLint transparent_loc = glGetUniformLocation(m_texture_shader, "transparent_background");
    GLint svg_loc = glGetUniformLocation(m_texture_shader, "svg_source");
    glUniform1i(transparent_loc, 0);
    glUniform1i(svg_loc, 1);  // Use SVG-style rendering
    
    // Bind SVG texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    GLint tex_loc = glGetUniformLocation(m_texture_shader, "in_texture");
    glUniform1i(tex_loc, 0);
    
    // FIXED: Use actual bed dimensions, not SVG dimensions!
    // MK4 bed is 250×210mm, use that for proper scaling
    float bed_width = 250.0f;   // MK4 bed width
    float bed_height = 210.0f;  // MK4 bed height  
    float half_w = bed_width / 2.0f;
    float half_h = bed_height / 2.0f;
    
    // Create flat quad vertices at Z=0 with FIXED texture coordinates
    float flat_surface[] = {
        // Position (x, y, z)     // Texture coords (u, v) - FIXED orientation
        -half_w, -half_h, 0.0f,   0.0f, 1.0f,  // Bottom-left  -> top-left in texture
         half_w, -half_h, 0.0f,   1.0f, 1.0f,  // Bottom-right -> top-right in texture
         half_w,  half_h, 0.0f,   1.0f, 0.0f,  // Top-right    -> bottom-right in texture
        -half_w,  half_h, 0.0f,   0.0f, 0.0f   // Top-left     -> bottom-left in texture
    };
    
    unsigned int flat_indices[] = {0, 1, 2, 2, 3, 0};
    
    // Create temporary VAO/VBO for flat surface
    GLuint temp_vao, temp_vbo, temp_ebo;
    glGenVertexArrays(1, &temp_vao);
    glGenBuffers(1, &temp_vbo);
    glGenBuffers(1, &temp_ebo);
    
    glBindVertexArray(temp_vao);
    glBindBuffer(GL_ARRAY_BUFFER, temp_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(flat_surface), flat_surface, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, temp_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(flat_indices), flat_indices, GL_STATIC_DRAW);
    
    // Set vertex attributes for printbed shader (position at 0, texcoords at 1)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Render flat textured surface
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    
    // Cleanup temporary buffers
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &temp_vao);
    glDeleteBuffers(1, &temp_vbo);
    glDeleteBuffers(1, &temp_ebo);
    
    glUseProgram(0);
}

void BedRenderer::renderGrid(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    glUseProgram(m_grid_shader);
    
    GLint view_loc = glGetUniformLocation(m_grid_shader, "view_model_matrix");
    GLint proj_loc = glGetUniformLocation(m_grid_shader, "projection_matrix");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Set grid color (more subtle, matching main slicer's grid colors)
    GLint color_loc = glGetUniformLocation(m_grid_shader, "uniform_color");
    if (m_has_model) {
        // Solid grid for system beds (DEFAULT_SOLID_GRID_COLOR)
        glUniform4f(color_loc, 0.9f, 0.9f, 0.9f, 0.6f);
    } else {
        // Transparent grid for custom beds (DEFAULT_TRANSPARENT_GRID_COLOR)  
        glUniform4f(color_loc, 0.9f, 0.9f, 0.9f, 0.4f);
    }
    
    glBindVertexArray(m_grid_vao);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, m_grid_vertex_count);
    glBindVertexArray(0);
    
    glUseProgram(0);
}

void BedRenderer::renderTextureWithGrid(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    // Render background surface first (like main slicer's render_default)
    glUseProgram(m_grid_shader);  // Use flat shader for simple background
    
    GLint view_loc = glGetUniformLocation(m_grid_shader, "view_model_matrix");
    GLint proj_loc = glGetUniformLocation(m_grid_shader, "projection_matrix");
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Set background color (matching main slicer's DEFAULT_MODEL_COLOR)
    GLint color_loc = glGetUniformLocation(m_grid_shader, "uniform_color");
    glUniform4f(color_loc, 0.8f, 0.8f, 0.8f, 1.0f);
    
    glDepthMask(GL_FALSE);  // Don't write to depth buffer for background
    glBindVertexArray(m_triangle_vao);
    glDrawElements(GL_TRIANGLES, m_triangle_index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    
    glUseProgram(0);
    
    // Then render grid on top
    renderGrid(view_matrix, projection_matrix);
}

void BedRenderer::cleanup() {
    if (m_model_vao) { glDeleteVertexArrays(1, &m_model_vao); m_model_vao = 0; }
    if (m_model_vbo) { glDeleteBuffers(1, &m_model_vbo); m_model_vbo = 0; }
    if (m_model_ebo) { glDeleteBuffers(1, &m_model_ebo); m_model_ebo = 0; }
    if (m_triangle_vao) { glDeleteVertexArrays(1, &m_triangle_vao); m_triangle_vao = 0; }
    if (m_triangle_vbo) { glDeleteBuffers(1, &m_triangle_vbo); m_triangle_vbo = 0; }
    if (m_triangle_ebo) { glDeleteBuffers(1, &m_triangle_ebo); m_triangle_ebo = 0; }
    if (m_grid_vao) { glDeleteVertexArrays(1, &m_grid_vao); m_grid_vao = 0; }
    if (m_grid_vbo) { glDeleteBuffers(1, &m_grid_vbo); m_grid_vbo = 0; }
    if (m_texture_id) { glDeleteTextures(1, &m_texture_id); m_texture_id = 0; }
    if (m_model_shader) { glDeleteProgram(m_model_shader); m_model_shader = 0; }
    if (m_texture_shader) { glDeleteProgram(m_texture_shader); m_texture_shader = 0; }
    if (m_grid_shader) { glDeleteProgram(m_grid_shader); m_grid_shader = 0; }
    
    m_initialized = false;
    m_has_model = false;
}

bool BedRenderer::loadSTLModel(const std::string& stl_path) {
    std::ifstream file(stl_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open STL file: " << stl_path << std::endl;
        return false;
    }
    
    m_model_vertices.clear();
    m_model_indices.clear();
    
    // Read header (80 bytes)
    char header[80];
    file.read(header, 80);
    
    // Read number of triangles (4 bytes)
    uint32_t num_triangles;
    file.read(reinterpret_cast<char*>(&num_triangles), 4);
    
    std::cout << "Loading STL with " << num_triangles << " triangles" << std::endl;
    
    // First pass: read all geometry and calculate bounds
    std::vector<float> temp_positions;
    std::vector<float> temp_normals;
    
    // Track bounds for proper positioning
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    float min_z = FLT_MAX, max_z = -FLT_MAX;
    
    // Read each triangle and store data
    for (uint32_t i = 0; i < num_triangles; i++) {
        // Read normal vector (3 floats)
        float normal[3];
        file.read(reinterpret_cast<char*>(normal), 12);
        
        // Read 3 vertices (9 floats)
        float vertices[9];
        file.read(reinterpret_cast<char*>(vertices), 36);
        
        // Read attribute byte count (2 bytes)
        uint16_t attr_count;
        file.read(reinterpret_cast<char*>(&attr_count), 2);
        
        // Store vertices and calculate bounds
        for (int v = 0; v < 3; v++) {
            float x = vertices[v*3 + 0];
            float y = vertices[v*3 + 1];
            float z = vertices[v*3 + 2];
            
            // Store position and normal
            temp_positions.push_back(x);
            temp_positions.push_back(y);
            temp_positions.push_back(z);
            temp_normals.push_back(normal[0]);
            temp_normals.push_back(normal[1]);
            temp_normals.push_back(normal[2]);
            
            // Update bounds
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
    }
    
    // Second pass: create vertices with correct texture coordinates
    float width = max_x - min_x;
    float height = max_y - min_y;
    
    for (size_t i = 0; i < temp_positions.size(); i += 3) {
        float x = temp_positions[i];
        float y = temp_positions[i + 1];
        float z = temp_positions[i + 2];
        
        // Position
        m_model_vertices.push_back(x);
        m_model_vertices.push_back(y);
        m_model_vertices.push_back(z);
        
        // Normal
        m_model_vertices.push_back(temp_normals[i]);
        m_model_vertices.push_back(temp_normals[i + 1]);
        m_model_vertices.push_back(temp_normals[i + 2]);
        
        // Texture coordinates (proper centering, not stretching like PrusaSlicer)
        float u, v;
        if (m_svg_width_mm > 0.0f && m_svg_height_mm > 0.0f) {
            // Use SVG's real-world dimensions for proper scaling
            // Center the texture on the bed instead of stretching to fit
            float bed_center_x = (min_x + max_x) / 2.0f;
            float bed_center_y = (min_y + max_y) / 2.0f;
            
            // Calculate UV based on distance from bed center using SVG's real size
            float u_offset = (x - bed_center_x) / m_svg_width_mm;
            float v_offset = (y - bed_center_y) / m_svg_height_mm;
            
            // Center texture: 0.5 is center, offset from there
            u = 0.5f + u_offset;
            v = 0.5f - v_offset; // Flip V to fix orientation
        } else {
            // Fallback to old method if SVG dimensions not available
            u = (x - min_x) / width;
            v = 1.0f - (y - min_y) / height;
        }
        
        m_model_vertices.push_back(u);
        m_model_vertices.push_back(v);
        
        // Add index
        m_model_indices.push_back(m_model_indices.size());
    }
    
    m_model_vertex_count = m_model_vertices.size() / 8;  // 8 floats per vertex
    m_model_index_count = m_model_indices.size();
    
    // Calculate model offset (like PrusaSlicer's approach)
    // Center the model and move it slightly down to avoid Z-fighting with texture surface
    m_model_offset_x = -(min_x + max_x) / 2.0f;  // Center X
    m_model_offset_y = -(min_y + max_y) / 2.0f;  // Center Y
    m_model_offset_z = -0.03f;  // Move down slightly (like PrusaSlicer's -0.03 offset)
    
    std::cout << "Model loaded, bounds: X[" << min_x << "," << max_x << "] Y[" << min_y << "," << max_y << "] Z[" << min_z << "," << max_z << "]" << std::endl;
    
    return m_model_vertex_count > 0;
}

void BedRenderer::setBedShape(const std::vector<std::pair<float, float>>& bed_shape) {
    m_bed_shape = bed_shape;
    std::cout << "Set bed shape with " << m_bed_shape.size() << " points" << std::endl;
}

void BedRenderer::createBedTriangles() {
    m_triangle_vertices.clear();
    m_triangle_indices.clear();
    
    if (m_bed_shape.empty()) {
        // Default to rectangular bed surface if no shape is defined
        float bed_size = 250.0f;  // 250mm bed
        float half_size = bed_size / 2.0f;
        
        // Create quad vertices with proper texture coordinates
        std::vector<float> quad_verts = {
            // Position (x, y, z)     // Texture coords (u, v)
            -half_size, -half_size, 0.0f,  0.0f, 0.0f,  // Bottom-left
             half_size, -half_size, 0.0f,  1.0f, 0.0f,  // Bottom-right
             half_size,  half_size, 0.0f,  1.0f, 1.0f,  // Top-right
            -half_size,  half_size, 0.0f,  0.0f, 1.0f   // Top-left
        };
        
        m_triangle_vertices = quad_verts;
        m_triangle_indices = {0, 1, 2, 2, 3, 0};
        
        m_triangle_vertex_count = 4;
        m_triangle_index_count = 6;
        return;
    }
    
    // Use dynamic bed shape - triangulate the polygon
    std::cout << "Creating bed triangles from " << m_bed_shape.size() << " shape points" << std::endl;
    
    // Calculate bounding box for texture coordinate mapping
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    
    for (const auto& point : m_bed_shape) {
        min_x = std::min(min_x, point.first);
        max_x = std::max(max_x, point.first);
        min_y = std::min(min_y, point.second);
        max_y = std::max(max_y, point.second);
    }
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    
    if (width <= 0.0f || height <= 0.0f) {
        std::cerr << "Invalid bed shape bounds" << std::endl;
        return;
    }
    
    // Simple triangulation for convex polygons (ear clipping for general case)
    triangulatePolygon(m_bed_shape, min_x, min_y, width, height);
    
    std::cout << "Generated " << m_triangle_vertex_count << " vertices and " << m_triangle_index_count << " indices for bed" << std::endl;
}

void BedRenderer::createProfessionalGrid() {
    // Create professional grid system like main slicer
    m_grid_vertices.clear();
    
    float grid_spacing = 10.0f;  // 10mm grid
    float z_offset = 0.01f;  // Slightly above bed surface
    
    if (m_bed_shape.empty()) {
        // Default rectangular grid
        float bed_size = 250.0f;
        float half_size = bed_size / 2.0f;
        
        // Vertical lines
        for (float x = -half_size; x <= half_size; x += grid_spacing) {
            m_grid_vertices.push_back(x);
            m_grid_vertices.push_back(-half_size);
            m_grid_vertices.push_back(z_offset);
            
            m_grid_vertices.push_back(x);
            m_grid_vertices.push_back(half_size);
            m_grid_vertices.push_back(z_offset);
        }
        
        // Horizontal lines
        for (float y = -half_size; y <= half_size; y += grid_spacing) {
            m_grid_vertices.push_back(-half_size);
            m_grid_vertices.push_back(y);
            m_grid_vertices.push_back(z_offset);
            
            m_grid_vertices.push_back(half_size);
            m_grid_vertices.push_back(y);
            m_grid_vertices.push_back(z_offset);
        }
    } else {
        // Dynamic grid based on bed shape bounds
        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;
        
        for (const auto& point : m_bed_shape) {
            min_x = std::min(min_x, point.first);
            max_x = std::max(max_x, point.first);
            min_y = std::min(min_y, point.second);
            max_y = std::max(max_y, point.second);
        }
        
        // Add some padding
        float padding = grid_spacing;
        min_x -= padding;
        max_x += padding;
        min_y -= padding;
        max_y += padding;
        
        // Vertical lines
        for (float x = floorf(min_x / grid_spacing) * grid_spacing; x <= max_x; x += grid_spacing) {
            m_grid_vertices.push_back(x);
            m_grid_vertices.push_back(min_y);
            m_grid_vertices.push_back(z_offset);
            
            m_grid_vertices.push_back(x);
            m_grid_vertices.push_back(max_y);
            m_grid_vertices.push_back(z_offset);
        }
        
        // Horizontal lines
        for (float y = floorf(min_y / grid_spacing) * grid_spacing; y <= max_y; y += grid_spacing) {
            m_grid_vertices.push_back(min_x);
            m_grid_vertices.push_back(y);
            m_grid_vertices.push_back(z_offset);
            
            m_grid_vertices.push_back(max_x);
            m_grid_vertices.push_back(y);
            m_grid_vertices.push_back(z_offset);
        }
        
        std::cout << "Created dynamic grid for bed bounds: X[" << min_x << "," << max_x 
                  << "] Y[" << min_y << "," << max_y << "]" << std::endl;
    }
    
    m_grid_vertex_count = m_grid_vertices.size() / 3;
}

void BedRenderer::createProfessionalTexture() {
    // Create professional texture with radial gradient (like main slicer's SVG handling)
    const int tex_width = 512;
    const int tex_height = 512;
    std::vector<unsigned char> texture_data(tex_width * tex_height * 3);
    
    // Colors from main slicer's printbed shader
    const float back_color_dark[3] = {0.235f, 0.235f, 0.235f};
    const float back_color_light[3] = {0.365f, 0.365f, 0.365f};
    
    for (int y = 0; y < tex_height; y++) {
        for (int x = 0; x < tex_width; x++) {
            int idx = (y * tex_width + x) * 3;
            
            // Calculate radial gradient (matching main slicer's approach)
            float u = (float)x / (float)tex_width;
            float v = (float)y / (float)tex_height;
            float dist = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));
            
            // Smoothstep function like in main slicer
            float t = std::min(1.0f, std::max(0.0f, dist * 2.0f));
            t = t * t * (3.0f - 2.0f * t);  // smoothstep
            
            // Mix colors
            float r = back_color_light[0] * (1.0f - t) + back_color_dark[0] * t;
            float g = back_color_light[1] * (1.0f - t) + back_color_dark[1] * t;
            float b = back_color_light[2] * (1.0f - t) + back_color_dark[2] * t;
            
            // Add subtle grid pattern
            bool is_grid = (x % 64 < 2) || (y % 64 < 2);
            if (is_grid) {
                r += 0.05f;
                g += 0.05f;
                b += 0.05f;
            }
            
            // Add center cross (like Prusa logo area)
            if ((x > tex_width/2 - 4 && x < tex_width/2 + 4) || 
                (y > tex_height/2 - 4 && y < tex_height/2 + 4)) {
                r += 0.1f;
                g += 0.08f;
                b += 0.02f;
            }
            
            texture_data[idx + 0] = (unsigned char)(std::min(255.0f, r * 255.0f));
            texture_data[idx + 1] = (unsigned char)(std::min(255.0f, g * 255.0f));
            texture_data[idx + 2] = (unsigned char)(std::min(255.0f, b * 255.0f));
        }
    }
    
    // Create OpenGL texture
    glGenTextures(1, &m_texture_id);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool BedRenderer::createProceduralTexture() {
    // Alias for createProfessionalTexture for consistency with header
    createProfessionalTexture();
    return true;
}

bool BedRenderer::loadSVGTexture(const std::string& svg_path) {
    std::cout << "Loading real SVG texture: " << svg_path << std::endl;
    
    // Check if file exists
    if (!std::filesystem::exists(svg_path)) {
        std::cerr << "SVG file not found: " << svg_path << std::endl;
        return false;
    }
    
    // Parse SVG using NanoSVG (same as PrusaSlicer)
    NSVGimage* image = nsvgParseFromFile(svg_path.c_str(), "px", 96.0f);
    if (image == nullptr) {
        std::cerr << "Failed to parse SVG file: " << svg_path << std::endl;
        return false;
    }
    
    std::cout << "SVG parsed successfully: " << image->width << "x" << image->height << std::endl;
    
    // Calculate SVG real-world dimensions (like PrusaSlicer does)
    // SVG uses 96 DPI, convert to millimeters: pixels / 96 * 25.4
    float svg_width_mm = (image->width / 96.0f) * 25.4f;
    float svg_height_mm = (image->height / 96.0f) * 25.4f;
    
    std::cout << "SVG real-world size: " << svg_width_mm << "x" << svg_height_mm << "mm" << std::endl;
    
    // Store for UV coordinate calculation (proper centering, not stretching)
    m_svg_width_mm = svg_width_mm;
    m_svg_height_mm = svg_height_mm;
    
    // Keep original SVG aspect ratio (don't stretch)
    const int target_width = 512;  
    float aspect_ratio = image->height / image->width;
    int target_height = (int)(target_width * aspect_ratio + 0.5f);
    
    // Use 1:1 scale for better quality (no unnecessary scaling)
    float scale = 1.0f;
    int width = (int)image->width;
    int height = (int)image->height;
    
    std::cout << "Rasterizing SVG to: " << width << "x" << height << " (preserving original size)" << std::endl;
    
    // Create rasterizer (same as PrusaSlicer)
    NSVGrasterizer* rast = nsvgCreateRasterizer();
    if (rast == nullptr) {
        std::cerr << "Failed to create SVG rasterizer" << std::endl;
        nsvgDelete(image);
        return false;
    }
    
    // Rasterize SVG to RGBA data (same as PrusaSlicer)
    std::vector<unsigned char> data(width * height * 4, 0);
    nsvgRasterize(rast, image, 0, 0, scale, data.data(), width, height, width * 4);
    
    // Create OpenGL texture from rasterized data
    if (m_texture_id != 0) {
        glDeleteTextures(1, &m_texture_id);
    }
    m_texture_id = loadTextureFromData(data.data(), width, height, 4);
    
    // Cleanup (same as PrusaSlicer)
    nsvgDeleteRasterizer(rast);
    nsvgDelete(image);
    
    bool success = (m_texture_id != 0);
    if (success) {
        std::cout << "Successfully loaded SVG texture: " << svg_path << std::endl;
    } else {
        std::cerr << "Failed to create OpenGL texture from SVG data" << std::endl;
    }
    
    return success;
}

bool BedRenderer::loadImageTexture(const std::string& image_path) {
    std::cout << "Image texture loading not yet implemented: " << image_path << std::endl;
    return false;
}

GLuint BedRenderer::loadTextureFromData(const unsigned char* data, int width, int height, int channels) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    GLenum format = (channels == 3) ? GL_RGB : GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture_id;
}

// Removed fake texture creation methods - now using real SVG parsing!

bool BedRenderer::createShaders() {
    // Create professional shaders matching main slicer
    m_model_shader = createShaderProgram(getGouraudVertexShader(), getGouraudFragmentShader());
    if (m_model_shader == 0) {
        std::cerr << "Failed to create model shader (gouraud)" << std::endl;
        return false;
    }
    
    m_texture_shader = createShaderProgram(getPrintbedVertexShader(), getPrintbedFragmentShader());
    if (m_texture_shader == 0) {
        std::cerr << "Failed to create texture shader (printbed)" << std::endl;
        return false;
    }
    
    m_grid_shader = createShaderProgram(getFlatVertexShader(), getFlatFragmentShader());
    if (m_grid_shader == 0) {
        std::cerr << "Failed to create grid shader (flat)" << std::endl;
        return false;
    }
    
    return true;
}

void BedRenderer::setupModelBuffers() {
    if (m_model_vertices.empty()) return;
    
    glGenVertexArrays(1, &m_model_vao);
    glGenBuffers(1, &m_model_vbo);
    glGenBuffers(1, &m_model_ebo);
    
    glBindVertexArray(m_model_vao);
    
    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, m_model_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_model_vertices.size() * sizeof(float), m_model_vertices.data(), GL_STATIC_DRAW);
    
    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_model_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_model_indices.size() * sizeof(unsigned int), m_model_indices.data(), GL_STATIC_DRAW);
    
    // Set vertex attributes to match printbed shader expectations:
    // - Location 0: position (3 floats)
    // - Location 1: texcoords (2 floats) - SKIP normals for texture shader
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Skip normal data and go directly to texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void BedRenderer::setupTriangleBuffers() {
    if (m_triangle_vertices.empty()) return;
    
    glGenVertexArrays(1, &m_triangle_vao);
    glGenBuffers(1, &m_triangle_vbo);
    glGenBuffers(1, &m_triangle_ebo);
    
    glBindVertexArray(m_triangle_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_triangle_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_triangle_vertices.size() * sizeof(float), m_triangle_vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_triangle_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_triangle_indices.size() * sizeof(unsigned int), m_triangle_indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void BedRenderer::setupGridBuffers() {
    if (m_grid_vertices.empty()) return;
    
    glGenVertexArrays(1, &m_grid_vao);
    glGenBuffers(1, &m_grid_vbo);
    
    glBindVertexArray(m_grid_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_grid_vertices.size() * sizeof(float), m_grid_vertices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

// Professional shader code matching main slicer
const char* BedRenderer::getPrintbedVertexShader() {
    return R"(
#version 330 core
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec2 v_tex_coord;

uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;

out vec2 tex_coord;

void main() {
    tex_coord = v_tex_coord;
    gl_Position = projection_matrix * view_model_matrix * vec4(v_position, 1.0);
}
)";
}

const char* BedRenderer::getPrintbedFragmentShader() {
    return R"(
#version 330 core
in vec2 tex_coord;

uniform sampler2D in_texture;
uniform bool transparent_background;
uniform bool svg_source;

out vec4 FragColor;

const vec3 back_color_dark  = vec3(0.1, 0.1, 0.1);
const vec3 back_color_light = vec3(0.15, 0.15, 0.15);

vec4 svg_color() {
    // takes foreground from texture
    vec4 fore_color = texture(in_texture, tex_coord);
    
    // calculates radial gradient
    vec3 back_color = vec3(mix(back_color_light, back_color_dark, 
                             smoothstep(0.0, 0.5, length(abs(tex_coord.xy) - vec2(0.5)))));
    
    // blends foreground with background
    return vec4(mix(back_color, fore_color.rgb, fore_color.a), 
                transparent_background ? fore_color.a : 1.0);
}

vec4 non_svg_color() {
    vec4 color = texture(in_texture, tex_coord);
    return vec4(color.rgb, transparent_background ? color.a * 0.25 : color.a);
}

void main() {
    vec4 color = svg_source ? svg_color() : non_svg_color();
    color.a = transparent_background ? color.a * 0.5 : color.a;
    FragColor = color;
}
)";
}

const char* BedRenderer::getGouraudVertexShader() {
    return R"(
#version 330 core
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_tex_coord;

uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;

// Professional lighting constants matching main slicer
#define INTENSITY_CORRECTION 0.6
const vec3 LIGHT_TOP_DIR = vec3(-0.4574957, 0.4574957, 0.7624929);
#define LIGHT_TOP_DIFFUSE    (0.8 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SPECULAR   (0.125 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SHININESS  20.0

const vec3 LIGHT_FRONT_DIR = vec3(0.6985074, 0.1397015, 0.6985074);
#define LIGHT_FRONT_DIFFUSE  (0.3 * INTENSITY_CORRECTION)
#define INTENSITY_AMBIENT    0.3

out vec2 intensity;
out vec2 tex_coord;

void main() {
    // Transform normal to eye space
    vec3 eye_normal = normalize(mat3(view_model_matrix) * v_normal);
    
    // Calculate lighting
    float NdotL = max(dot(eye_normal, LIGHT_TOP_DIR), 0.0);
    intensity.x = INTENSITY_AMBIENT + NdotL * LIGHT_TOP_DIFFUSE;
    
    vec4 position = view_model_matrix * vec4(v_position, 1.0);
    intensity.y = LIGHT_TOP_SPECULAR * pow(max(dot(-normalize(position.xyz), 
                                                  reflect(-LIGHT_TOP_DIR, eye_normal)), 0.0), 
                                           LIGHT_TOP_SHININESS);
    
    // Second light source
    NdotL = max(dot(eye_normal, LIGHT_FRONT_DIR), 0.0);
    intensity.x += NdotL * LIGHT_FRONT_DIFFUSE;
    
    tex_coord = v_tex_coord;
    gl_Position = projection_matrix * position;
}
)";
}

const char* BedRenderer::getGouraudFragmentShader() {
    return R"(
#version 330 core
in vec2 intensity;
in vec2 tex_coord;

uniform vec4 uniform_color;

out vec4 FragColor;

void main() {
    FragColor = vec4(vec3(intensity.y) + uniform_color.rgb * intensity.x, uniform_color.a);
}
)";
}

const char* BedRenderer::getFlatVertexShader() {
    return R"(
#version 330 core
layout (location = 0) in vec3 v_position;

uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;

void main() {
    gl_Position = projection_matrix * view_model_matrix * vec4(v_position, 1.0);
}
)";
}

const char* BedRenderer::getFlatFragmentShader() {
    return R"(
#version 330 core
uniform vec4 uniform_color;

out vec4 FragColor;

void main() {
    FragColor = uniform_color;
}
)";
}

GLuint BedRenderer::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed: " << info_log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint BedRenderer::createShaderProgram(const char* vertex_source, const char* fragment_source) {
    GLuint vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_source);
    
    if (vertex_shader == 0 || fragment_shader == 0) {
        if (vertex_shader) glDeleteShader(vertex_shader);
        if (fragment_shader) glDeleteShader(fragment_shader);
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "Shader program linking failed: " << info_log << std::endl;
        glDeleteProgram(program);
        program = 0;
    }
    
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    return program;
}

void BedRenderer::triangulatePolygon(const std::vector<std::pair<float, float>>& shape, 
                                     float min_x, float min_y, float width, float height) {
    if (shape.size() < 3) {
        std::cerr << "Cannot triangulate polygon with less than 3 vertices" << std::endl;
        return;
    }
    
    // Simple fan triangulation - works for convex polygons and many concave ones
    // For more complex shapes, we'd need ear clipping or other advanced algorithms
    
    // Add vertices with texture coordinates
    for (size_t i = 0; i < shape.size(); ++i) {
        float x = shape[i].first;
        float y = shape[i].second;
        float z = 0.0f; // Bed is at Z=0
        
        // Calculate texture coordinates (0.0 to 1.0)
        float u = (x - min_x) / width;
        float v = (y - min_y) / height;
        
        // Position (x, y, z)
        m_triangle_vertices.push_back(x);
        m_triangle_vertices.push_back(y);
        m_triangle_vertices.push_back(z);
        
        // Texture coordinates (u, v)
        m_triangle_vertices.push_back(u);
        m_triangle_vertices.push_back(v);
    }
    
    // Create triangle indices using fan triangulation
    // Connect all triangles to vertex 0
    for (size_t i = 1; i < shape.size() - 1; ++i) {
        m_triangle_indices.push_back(0);       // First vertex
        m_triangle_indices.push_back(i);       // Current vertex
        m_triangle_indices.push_back(i + 1);   // Next vertex
    }
    
    m_triangle_vertex_count = shape.size();
    m_triangle_index_count = m_triangle_indices.size();
    
    std::cout << "Triangulated bed: " << m_triangle_vertex_count << " vertices, " 
              << (m_triangle_index_count / 3) << " triangles" << std::endl;
} 