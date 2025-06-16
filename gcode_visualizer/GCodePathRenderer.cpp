#include "GCodePathRenderer.hpp"
#include <iostream>
#include <cstring>
#include <cmath>

// Simple matrix type (copy from GCodeVisualizerApp.hpp to avoid include conflicts)
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

// Vertex shader for G-code paths
const char* path_vertex_shader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vertexColor;

void main() {
    gl_Position = uProjection * uView * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)";

// Fragment shader for G-code paths
const char* path_fragment_shader = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

GCodePathRenderer::GCodePathRenderer() {
}

GCodePathRenderer::~GCodePathRenderer() {
    cleanup();
}

bool GCodePathRenderer::initialize() {
    std::cout << "Initializing G-code path renderer..." << std::endl;
    
    // Create shader program
    if (!createShaders()) {
        std::cerr << "Failed to create path shaders" << std::endl;
        return false;
    }
    
    // Generate vertex array and buffer objects
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    
    // Set up vertex attributes
    // Position (x, y, z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    std::cout << "G-code path renderer initialized successfully" << std::endl;
    return true;
}

void GCodePathRenderer::setGCodeMoves(const std::vector<SimpleGCode::GCodeMove>& moves) {
    std::cout << "Loading " << moves.size() << " G-code moves for rendering..." << std::endl;
    
    generatePathGeometry(moves);
    
    // Upload vertex data to GPU
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(PathVertex), m_vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    m_vertex_count = m_vertices.size();
    std::cout << "Generated " << m_vertex_count << " vertices for G-code paths" << std::endl;
}

void GCodePathRenderer::render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    if (m_vertex_count == 0) {
        return;  // No paths to render
    }
    
    glUseProgram(m_shader_program);
    
    // Set uniforms
    GLint view_loc = glGetUniformLocation(m_shader_program, "uView");
    GLint proj_loc = glGetUniformLocation(m_shader_program, "uProjection");
    
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    
    // Render paths as lines
    glBindVertexArray(m_vao);
    glLineWidth(2.0f);  // Make lines a bit thicker
    glDrawArrays(GL_LINES, 0, m_vertex_count);
    glBindVertexArray(0);
    
    glUseProgram(0);
}

void GCodePathRenderer::cleanup() {
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_shader_program) {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    
    m_vertices.clear();
    m_vertex_count = 0;
}

bool GCodePathRenderer::createShaders() {
    // Compile vertex shader
    GLuint vertex_shader = compileShader(path_vertex_shader, GL_VERTEX_SHADER);
    if (vertex_shader == 0) {
        return false;
    }
    
    // Compile fragment shader
    GLuint fragment_shader = compileShader(path_fragment_shader, GL_FRAGMENT_SHADER);
    if (fragment_shader == 0) {
        glDeleteShader(vertex_shader);
        return false;
    }
    
    // Create shader program
    m_shader_program = glCreateProgram();
    glAttachShader(m_shader_program, vertex_shader);
    glAttachShader(m_shader_program, fragment_shader);
    glLinkProgram(m_shader_program);
    
    // Check for linking errors
    GLint success;
    glGetProgramiv(m_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(m_shader_program, 512, nullptr, info_log);
        std::cerr << "Shader program linking failed: " << info_log << std::endl;
        
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
        return false;
    }
    
    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    std::cout << "Path shader program created successfully" << std::endl;
    return true;
}

GLuint GCodePathRenderer::compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed (" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") 
                  << "): " << info_log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

void GCodePathRenderer::generatePathGeometry(const std::vector<SimpleGCode::GCodeMove>& moves) {
    m_vertices.clear();
    m_vertices.reserve(moves.size() * 2);  // Each move creates a line (2 vertices)
    
    for (const auto& move : moves) {
        // Skip moves that don't actually move anywhere
        float dx = move.end_pos.x - move.start_pos.x;
        float dy = move.end_pos.y - move.start_pos.y;
        float dz = move.end_pos.z - move.start_pos.z;
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance < 0.001f) {
            continue;  // Skip tiny moves
        }
        
        // Get color based on move type
        float r, g, b;
        getMoveColor(move.type, r, g, b);
        
        // Create line from start to end position
        // Transform G-code coordinates to match bed model coordinate system
        // G-code uses front-left corner as (0,0), but bed model is centered around (0,0)
        // For Prusa MK4: bed model spans -127 to +127 (254mm), -129 to +140 (269mm)
        // Standard Prusa printable area is roughly 250x210mm starting from (0,0)
        // We need to offset G-code coordinates to center them on the bed model
        
        PathVertex start_vertex;
        start_vertex.x = move.start_pos.x - 125.0f;  // Center X: shift by half of 250mm
        start_vertex.y = move.start_pos.y - 105.0f;  // Center Y: shift by half of 210mm  
        start_vertex.z = move.start_pos.z;
        start_vertex.r = r;
        start_vertex.g = g;
        start_vertex.b = b;
        
        PathVertex end_vertex;
        end_vertex.x = move.end_pos.x - 125.0f;      // Center X: shift by half of 250mm
        end_vertex.y = move.end_pos.y - 105.0f;      // Center Y: shift by half of 210mm
        end_vertex.z = move.end_pos.z;
        end_vertex.r = r;
        end_vertex.g = g;
        end_vertex.b = b;
        
        m_vertices.push_back(start_vertex);
        m_vertices.push_back(end_vertex);
    }
}

void GCodePathRenderer::getMoveColor(SimpleGCode::MoveType type, float& r, float& g, float& b) {
    switch (type) {
        case SimpleGCode::MoveType::Extrusion:
            // Blue for extrusion moves (actual printing)
            r = 0.2f; g = 0.6f; b = 1.0f;
            break;
        case SimpleGCode::MoveType::Travel:
            // Green for travel moves (non-printing)
            r = 0.2f; g = 1.0f; b = 0.2f;
            break;
        case SimpleGCode::MoveType::Retraction:
            // Red for retractions
            r = 1.0f; g = 0.2f; b = 0.2f;
            break;
        case SimpleGCode::MoveType::Unretraction:
            // Yellow for unretractions
            r = 1.0f; g = 1.0f; b = 0.2f;
            break;
        default:
            // White for unknown moves
            r = 1.0f; g = 1.0f; b = 1.0f;
            break;
    }
} 