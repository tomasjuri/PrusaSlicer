#include "BedRenderer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>

// Note: STB_IMAGE_WRITE_IMPLEMENTATION is already defined in ImageExporter.cpp

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
    std::cout << "Initializing advanced bed renderer..." << std::endl;
    
    m_stl_path = stl_path;
    m_svg_path = svg_path;
    
    // Create shaders first
    if (!createShaders()) {
        std::cerr << "Failed to create bed shaders" << std::endl;
        return false;
    }
    
    // Load STL model if provided
    if (!stl_path.empty()) {
        std::cout << "Loading bed model: " << stl_path << std::endl;
        if (loadSTLModel(stl_path)) {
            m_has_model = true;
            setupModelBuffers();
            std::cout << "Loaded bed model with " << m_model_vertices.size()/6 << " vertices" << std::endl;
        } else {
            std::cerr << "Failed to load STL model, falling back to simple bed" << std::endl;
        }
    }
    
    // Load SVG texture if provided
    if (!svg_path.empty()) {
        std::cout << "Loading bed texture: " << svg_path << std::endl;
        if (loadSVGTexture(svg_path)) {
            m_has_texture = true;
            std::cout << "Loaded bed texture successfully" << std::endl;
        } else {
            std::cerr << "Failed to load SVG texture" << std::endl;
        }
    }
    
    // Always create grid overlay
    createGridOverlay();
    setupGridBuffers();
    
    m_initialized = true;
    std::cout << "Bed renderer initialized: model=" << (m_has_model ? "yes" : "no") 
              << ", texture=" << (m_has_texture ? "yes" : "no") << std::endl;
    
    return true;
}

void BedRenderer::render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    if (!m_initialized) return;
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Render bed model
    if (m_has_model && m_model_index_count > 0) {
        glUseProgram(m_model_shader);
        
        // Set uniforms
        GLint view_loc = glGetUniformLocation(m_model_shader, "view_matrix");
        GLint proj_loc = glGetUniformLocation(m_model_shader, "projection_matrix");
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
        
        // Set texture if available
        if (m_has_texture) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_texture_id);
            GLint tex_loc = glGetUniformLocation(m_model_shader, "bedTexture");
            glUniform1i(tex_loc, 0);
            GLint use_tex_loc = glGetUniformLocation(m_model_shader, "useTexture");
            glUniform1i(use_tex_loc, 1);
        } else {
            GLint use_tex_loc = glGetUniformLocation(m_model_shader, "useTexture");
            glUniform1i(use_tex_loc, 0);
        }
        
        // Render model
        glBindVertexArray(m_model_vao);
        glDrawElements(GL_TRIANGLES, m_model_index_count, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glUseProgram(0);
    }
    
    // Render grid overlay
    if (m_grid_vertex_count > 0) {
        glUseProgram(m_grid_shader);
        
        GLint view_loc = glGetUniformLocation(m_grid_shader, "view_matrix");
        GLint proj_loc = glGetUniformLocation(m_grid_shader, "projection_matrix");
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
        
        glBindVertexArray(m_grid_vao);
        glLineWidth(1.0f);
        glDrawArrays(GL_LINES, 0, m_grid_vertex_count);
        glBindVertexArray(0);
        
        glUseProgram(0);
    }
    
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
}

void BedRenderer::cleanup() {
    if (m_model_vao) { glDeleteVertexArrays(1, &m_model_vao); m_model_vao = 0; }
    if (m_model_vbo) { glDeleteBuffers(1, &m_model_vbo); m_model_vbo = 0; }
    if (m_model_ebo) { glDeleteBuffers(1, &m_model_ebo); m_model_ebo = 0; }
    if (m_grid_vao) { glDeleteVertexArrays(1, &m_grid_vao); m_grid_vao = 0; }
    if (m_grid_vbo) { glDeleteBuffers(1, &m_grid_vbo); m_grid_vbo = 0; }
    if (m_texture_id) { glDeleteTextures(1, &m_texture_id); m_texture_id = 0; }
    if (m_model_shader) { glDeleteProgram(m_model_shader); m_model_shader = 0; }
    if (m_grid_shader) { glDeleteProgram(m_grid_shader); m_grid_shader = 0; }
    
    m_initialized = false;
    m_has_model = false;
    m_has_texture = false;
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
    
    std::cout << "Loading binary STL with " << num_triangles << " triangles" << std::endl;
    
    // Track model bounds for proper UV mapping
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    float min_z = FLT_MAX, max_z = -FLT_MAX;
    
    // Read each triangle
    for (uint32_t i = 0; i < num_triangles; i++) {
        // Read normal vector (3 floats)
        float normal[3];
        file.read(reinterpret_cast<char*>(normal), 12);
        
        // Read 3 vertices (9 floats)
        float vertices[9];
        file.read(reinterpret_cast<char*>(vertices), 36);
        
        // Read attribute byte count (2 bytes, usually 0)
        uint16_t attr_count;
        file.read(reinterpret_cast<char*>(&attr_count), 2);
        
        // Add vertices to our array and track bounds
        for (int v = 0; v < 3; v++) {
            float x = vertices[v*3 + 0];
            float y = vertices[v*3 + 1];
            float z = vertices[v*3 + 2];
            
            // Update bounds
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
            
            // Position
            m_model_vertices.push_back(x);
            m_model_vertices.push_back(y);
            m_model_vertices.push_back(z);
            // Normal
            m_model_vertices.push_back(normal[0]);
            m_model_vertices.push_back(normal[1]);
            m_model_vertices.push_back(normal[2]);
            
            // Add index
            m_model_indices.push_back(m_model_indices.size());
        }
    }
    
    m_model_vertex_count = m_model_vertices.size() / 6;  // 6 floats per vertex (pos + normal)
    m_model_index_count = m_model_indices.size();
    
    // Print actual model bounds for debugging
    std::cout << "STL Model bounds:" << std::endl;
    std::cout << "  X: " << min_x << " to " << max_x << " (size: " << (max_x - min_x) << ")" << std::endl;
    std::cout << "  Y: " << min_y << " to " << max_y << " (size: " << (max_y - min_y) << ")" << std::endl;
    std::cout << "  Z: " << min_z << " to " << max_z << " (size: " << (max_z - min_z) << ")" << std::endl;
    
    return m_model_vertex_count > 0;
}

void BedRenderer::parseSTLTriangle(const std::string& line, std::vector<float>& vertices) {
    std::istringstream iss(line);
    std::string word;
    float x, y, z;
    
    if (iss >> word >> x >> y >> z && word == "vertex") {
        // Add vertex position
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
        // Add dummy normal (we'll fix this later if needed)
        vertices.push_back(0.0f);
        vertices.push_back(0.0f);
        vertices.push_back(1.0f);
    }
}

bool BedRenderer::loadSVGTexture(const std::string& svg_path) {
    // Create a realistic Prusa-style bed texture
    const int tex_width = 512;
    const int tex_height = 512;
    std::vector<unsigned char> texture_data(tex_width * tex_height * 3);
    
    // Create a realistic Prusa MK4 bed texture
    for (int y = 0; y < tex_height; y++) {
        for (int x = 0; x < tex_width; x++) {
            int idx = (y * tex_width + x) * 3;
            
            // Base color - darker steel gray (more realistic for Prusa MK4)
            unsigned char base_r = 85;
            unsigned char base_g = 85; 
            unsigned char base_b = 90;  // Slightly bluer like steel
            
            // Add fine random texture to simulate steel surface
            float fine_noise = (std::sin(x * 0.3f) * std::cos(y * 0.3f)) * 8.0f;
            fine_noise += (std::sin(x * 0.05f) + std::sin(y * 0.05f)) * 5.0f;
            
            base_r = std::clamp((int)(base_r + fine_noise), 75, 105);
            base_g = std::clamp((int)(base_g + fine_noise), 75, 105);
            base_b = std::clamp((int)(base_b + fine_noise), 80, 110);
            
            // Add measurement grid every 50 pixels (represents ~25mm on actual bed)
            bool is_grid_line = (x % 50 < 1) || (y % 50 < 1);
            if (is_grid_line) {
                base_r = std::clamp(base_r + 25, 0, 255);  // Lighter grid lines
                base_g = std::clamp(base_g + 25, 0, 255);
                base_b = std::clamp(base_b + 25, 0, 255);
            }
            
            // Add major grid every 100 pixels (represents ~50mm)
            bool is_major_grid = (x % 100 < 2) || (y % 100 < 2);
            if (is_major_grid) {
                base_r = std::clamp(base_r + 15, 0, 255);
                base_g = std::clamp(base_g + 15, 0, 255);
                base_b = std::clamp(base_b + 15, 0, 255);
            }
            
            // Add center cross lines for origin reference (Prusa logo area)
            if ((x > tex_width/2 - 3 && x < tex_width/2 + 3) || 
                (y > tex_height/2 - 3 && y < tex_height/2 + 3)) {
                base_r = std::clamp(base_r + 30, 0, 255);
                base_g = std::clamp(base_g + 25, 0, 255);  // Slightly orange tint
                base_b = std::clamp(base_b + 10, 0, 255);
            }
            
            // Add corner markers
            if (((x < 20 || x > tex_width - 20) && (y < 20 || y > tex_height - 20))) {
                base_r = std::clamp(base_r + 20, 0, 255);
                base_g = std::clamp(base_g + 15, 0, 255);
                base_b = std::clamp(base_b + 5, 0, 255);
            }
            
            texture_data[idx + 0] = base_r;  // R
            texture_data[idx + 1] = base_g;  // G
            texture_data[idx + 2] = base_b;  // B
        }
    }
    
    // Create OpenGL texture
    glGenTextures(1, &m_texture_id);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

bool BedRenderer::loadPNGTexture(const std::string& png_path) {
    // Implementation for PNG loading would go here
    return false;
}

void BedRenderer::createGridOverlay() {
    m_grid_vertices.clear();
    
    // Create grid overlay matching STL model coordinate system
    // STL bounds: X: -127 to +127 (254mm), Y: -129 to +140 (269mm)
    float min_x = -127.0f;
    float max_x = 127.0f;
    float min_y = -129.0f;
    float max_y = 140.0f;
    float spacing = 10.0f;
    
    // Vertical lines (constant X, varying Y)
    for (float x = min_x; x <= max_x; x += spacing) {
        m_grid_vertices.push_back(x);
        m_grid_vertices.push_back(min_y);
        m_grid_vertices.push_back(0.1f);  // Slightly above the bed
        
        m_grid_vertices.push_back(x);
        m_grid_vertices.push_back(max_y);
        m_grid_vertices.push_back(0.1f);
    }
    
    // Horizontal lines (constant Y, varying X)
    for (float y = min_y; y <= max_y; y += spacing) {
        m_grid_vertices.push_back(min_x);
        m_grid_vertices.push_back(y);
        m_grid_vertices.push_back(0.1f);
        
        m_grid_vertices.push_back(max_x);
        m_grid_vertices.push_back(y);
        m_grid_vertices.push_back(0.1f);
    }
    
    m_grid_vertex_count = m_grid_vertices.size() / 3;
}

bool BedRenderer::createShaders() {
    // Create model shader
    m_model_shader = createShaderProgram(getModelVertexShader(), getModelFragmentShader());
    if (m_model_shader == 0) {
        std::cerr << "Failed to create model shader" << std::endl;
        return false;
    }
    
    // Create grid shader
    m_grid_shader = createShaderProgram(getGridVertexShader(), getGridFragmentShader());
    if (m_grid_shader == 0) {
        std::cerr << "Failed to create grid shader" << std::endl;
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
    
    // Set vertex attributes
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
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

const char* BedRenderer::getModelVertexShader() {
    return R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    FragPos = aPos;
    Normal = aNormal;
    
    // UV mapping using actual STL model bounds
    // X: -127 to +127 (254mm), Y: -129 to +140 (269mm)
    float u = (aPos.x + 127.0) / 254.0;  // Map X range to 0-1
    float v = (aPos.y + 129.0) / 269.0;  // Map Y range to 0-1
    
    // Use texture coordinates directly without scaling for proper alignment
    TexCoord = vec2(u, v);
    
    gl_Position = projection_matrix * view_matrix * vec4(aPos, 1.0);
}
)";
}

const char* BedRenderer::getModelFragmentShader() {
    return R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D bedTexture;
uniform bool useTexture;

out vec4 FragColor;

void main() {
    if (useTexture) {
        vec3 texColor = texture(bedTexture, TexCoord).rgb;
        
        // Enhanced lighting model for better contrast and realism
        vec3 lightDir1 = normalize(vec3(0.6, 0.6, 1.0));    // Primary light from above
        vec3 lightDir2 = normalize(vec3(-0.3, 0.3, 0.5));   // Secondary fill light
        vec3 normal = normalize(Normal);
        
        // Calculate diffuse lighting from multiple sources
        float diffuse1 = max(dot(normal, lightDir1), 0.0);
        float diffuse2 = max(dot(normal, lightDir2), 0.0) * 0.3;  // Weaker fill light
        float ambient = 0.4;  // Increased ambient for better visibility
        
        float totalLight = ambient + diffuse1 + diffuse2;
        totalLight = min(totalLight, 1.2);  // Prevent overexposure
        
        // Apply lighting and increase contrast
        vec3 finalColor = texColor * totalLight;
        
        // Slight contrast boost for better definition
        finalColor = pow(finalColor, vec3(0.9));  // Gamma adjustment
        
        FragColor = vec4(finalColor, 1.0);
    } else {
        // Medium gray color with lighting
        vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
        vec3 normal = normalize(Normal);
        float lightIntensity = max(dot(normal, lightDir), 0.4);
        vec3 bedColor = vec3(0.4, 0.4, 0.4);
        FragColor = vec4(bedColor * lightIntensity, 1.0);
    }
}
)";
}

const char* BedRenderer::getGridVertexShader() {
    return R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main() {
    gl_Position = projection_matrix * view_matrix * vec4(aPos, 1.0);
}
)";
}

const char* BedRenderer::getGridFragmentShader() {
    return R"(
#version 330 core
out vec4 FragColor;

void main() {
    // Light gray grid lines
    FragColor = vec4(0.7, 0.7, 0.7, 0.8);
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
        std::cerr << "Bed shader compilation failed: " << info_log << std::endl;
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
        std::cerr << "Bed shader program linking failed: " << info_log << std::endl;
        glDeleteProgram(program);
        program = 0;
    }
    
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    return program;
} 