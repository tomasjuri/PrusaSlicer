#include "BedRenderer.hpp"
#include "GCodeVisualizerApp.hpp"  // For Mat4x4 definition
#include <iostream>
#include <cmath>

BedRenderer::BedRenderer() {
}

BedRenderer::~BedRenderer() {
    cleanup();
}

void BedRenderer::initialize(float width, float height, float grid_spacing) {
    if (m_initialized) {
        cleanup();
    }
    
    m_width = width;
    m_height = height;
    m_grid_spacing = grid_spacing;
    
    std::cout << "Initializing bed renderer: " << width << "x" << height << "mm, grid: " << grid_spacing << "mm" << std::endl;
    
    // Create geometry
    createGridGeometry();
    createSurfaceGeometry();
    
    // Create shaders
    createShaders();
    
    // Setup OpenGL buffers
    setupGridBuffers();
    setupSurfaceBuffers();
    
    m_initialized = true;
}

void BedRenderer::render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    if (!m_initialized) {
        return;
    }
    
    glUseProgram(m_shader_program);
    
    // Set uniforms (simplified - in a full implementation you'd get uniform locations)
    // For now, we'll render with a simple approach
    
    // Enable blending for grid transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Render surface first (darker)
    glBindVertexArray(m_surface_vao);
    glDrawElements(GL_TRIANGLES, m_surface_index_count, GL_UNSIGNED_INT, 0);
    
    // Render grid lines (lighter)
    glBindVertexArray(m_grid_vao);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, m_grid_vertex_count);
    
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glUseProgram(0);
}

void BedRenderer::cleanup() {
    if (m_grid_vao) {
        glDeleteVertexArrays(1, &m_grid_vao);
        m_grid_vao = 0;
    }
    if (m_grid_vbo) {
        glDeleteBuffers(1, &m_grid_vbo);
        m_grid_vbo = 0;
    }
    if (m_surface_vao) {
        glDeleteVertexArrays(1, &m_surface_vao);
        m_surface_vao = 0;
    }
    if (m_surface_vbo) {
        glDeleteBuffers(1, &m_surface_vbo);
        m_surface_vbo = 0;
    }
    if (m_surface_ebo) {
        glDeleteBuffers(1, &m_surface_ebo);
        m_surface_ebo = 0;
    }
    if (m_shader_program) {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    
    m_initialized = false;
}

void BedRenderer::createGridGeometry() {
    m_grid_vertices.clear();
    
    // Create vertical lines
    for (float x = 0; x <= m_width; x += m_grid_spacing) {
        // Line from (x, 0, 0) to (x, height, 0)
        m_grid_vertices.push_back(x);
        m_grid_vertices.push_back(0.0f);
        m_grid_vertices.push_back(0.0f);
        
        m_grid_vertices.push_back(x);
        m_grid_vertices.push_back(m_height);
        m_grid_vertices.push_back(0.0f);
    }
    
    // Create horizontal lines
    for (float y = 0; y <= m_height; y += m_grid_spacing) {
        // Line from (0, y, 0) to (width, y, 0)
        m_grid_vertices.push_back(0.0f);
        m_grid_vertices.push_back(y);
        m_grid_vertices.push_back(0.0f);
        
        m_grid_vertices.push_back(m_width);
        m_grid_vertices.push_back(y);
        m_grid_vertices.push_back(0.0f);
    }
    
    m_grid_vertex_count = m_grid_vertices.size() / 3;
}

void BedRenderer::createSurfaceGeometry() {
    m_surface_vertices.clear();
    m_surface_indices.clear();
    
    // Create a simple rectangle for the bed surface
    // Bottom-left
    m_surface_vertices.push_back(0.0f);
    m_surface_vertices.push_back(0.0f);
    m_surface_vertices.push_back(-0.1f); // Slightly below Z=0
    
    // Bottom-right
    m_surface_vertices.push_back(m_width);
    m_surface_vertices.push_back(0.0f);
    m_surface_vertices.push_back(-0.1f);
    
    // Top-right
    m_surface_vertices.push_back(m_width);
    m_surface_vertices.push_back(m_height);
    m_surface_vertices.push_back(-0.1f);
    
    // Top-left
    m_surface_vertices.push_back(0.0f);
    m_surface_vertices.push_back(m_height);
    m_surface_vertices.push_back(-0.1f);
    
    // Create triangles (two triangles for the rectangle)
    m_surface_indices = {
        0, 1, 2,  // First triangle
        0, 2, 3   // Second triangle
    };
    
    m_surface_index_count = m_surface_indices.size();
}

void BedRenderer::createShaders() {
    const char* vertex_src = getVertexShaderSource();
    const char* fragment_src = getFragmentShaderSource();
    
    m_shader_program = createShaderProgram(vertex_src, fragment_src);
}

void BedRenderer::setupGridBuffers() {
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

void BedRenderer::setupSurfaceBuffers() {
    glGenVertexArrays(1, &m_surface_vao);
    glGenBuffers(1, &m_surface_vbo);
    glGenBuffers(1, &m_surface_ebo);
    
    glBindVertexArray(m_surface_vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_surface_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_surface_vertices.size() * sizeof(float), m_surface_vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surface_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_surface_indices.size() * sizeof(unsigned int), m_surface_indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

const char* BedRenderer::getVertexShaderSource() {
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

const char* BedRenderer::getFragmentShaderSource() {
    return R"(
#version 330 core
out vec4 FragColor;

void main() {
    // Light gray color for bed elements
    FragColor = vec4(0.8, 0.8, 0.8, 0.6);
}
)";
}

GLuint BedRenderer::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check compilation
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed: " << info_log << std::endl;
        return 0;
    }
    
    return shader;
}

GLuint BedRenderer::createShaderProgram(const char* vertex_source, const char* fragment_source) {
    GLuint vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_source);
    
    if (vertex_shader == 0 || fragment_shader == 0) {
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    
    // Check linking
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "Shader program linking failed: " << info_log << std::endl;
        return 0;
    }
    
    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    return program;
} 