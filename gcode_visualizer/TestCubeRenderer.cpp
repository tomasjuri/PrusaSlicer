#include "TestCubeRenderer.hpp"
#include "GCodeVisualizerApp.hpp"  // For Mat4x4 definition
#include <iostream>
#include <cmath>

TestCubeRenderer::TestCubeRenderer() {
}

TestCubeRenderer::~TestCubeRenderer() {
    cleanup();
}

void TestCubeRenderer::initialize() {
    if (m_initialized) {
        cleanup();
    }
    
    std::cout << "Initializing test cube renderer..." << std::endl;
    
    // Create cube geometry
    createCubeGeometry();
    
    // Create shaders
    createShaders();
    
    // Setup OpenGL buffers
    setupBuffers();
    
    m_initialized = true;
    std::cout << "Test cube renderer initialized successfully" << std::endl;
}

void TestCubeRenderer::render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
    if (!m_initialized) {
        std::cerr << "TestCubeRenderer not initialized!" << std::endl;
        return;
    }
    
    glUseProgram(m_shader_program);
    
    // Get uniform locations
    GLint view_loc = glGetUniformLocation(m_shader_program, "view_matrix");
    GLint proj_loc = glGetUniformLocation(m_shader_program, "projection_matrix");
    
    // Set uniforms
    if (view_loc >= 0) {
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix.data());
    }
    if (proj_loc >= 0) {
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.data());
    }
    
    // Render the cube
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    glUseProgram(0);
}

void TestCubeRenderer::cleanup() {
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_ebo) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }
    if (m_shader_program) {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    
    m_initialized = false;
}

void TestCubeRenderer::createCubeGeometry() {
    // Create a 10mm x 10mm x 10mm cube centered at (0, 0, 5) on the Prusa bed
    // Bed center is now at origin (0, 0), cube bottom at Z=0, top at Z=10
    float center_x = 0.0f;    // Bed center X (now at origin)
    float center_y = 0.0f;    // Bed center Y (now at origin)
    float size = 10.0f;       // 10mm cube
    float half_size = size / 2.0f;
    
    m_vertices = {
        // Bottom face (Z = 0)
        center_x - half_size, center_y - half_size, 0.0f,           // 0: bottom-left-front
        center_x + half_size, center_y - half_size, 0.0f,           // 1: bottom-right-front
        center_x + half_size, center_y + half_size, 0.0f,           // 2: bottom-right-back
        center_x - half_size, center_y + half_size, 0.0f,           // 3: bottom-left-back
        
        // Top face (Z = 10)
        center_x - half_size, center_y - half_size, size,           // 4: top-left-front
        center_x + half_size, center_y - half_size, size,           // 5: top-right-front
        center_x + half_size, center_y + half_size, size,           // 6: top-right-back
        center_x - half_size, center_y + half_size, size            // 7: top-left-back
    };
    
    m_indices = {
        // Bottom face
        0, 1, 2,  2, 3, 0,
        // Top face
        4, 7, 6,  6, 5, 4,
        // Front face
        0, 4, 5,  5, 1, 0,
        // Back face
        2, 6, 7,  7, 3, 2,
        // Left face
        0, 3, 7,  7, 4, 0,
        // Right face
        1, 5, 6,  6, 2, 1
    };
    
    std::cout << "Created test cube geometry: " << m_vertices.size()/3 << " vertices, " 
              << m_indices.size()/3 << " triangles" << std::endl;
    std::cout << "Cube position: (" << center_x << ", " << center_y << ", " << size/2.0f << ")" << std::endl;
}

void TestCubeRenderer::createShaders() {
    const char* vertex_src = getVertexShaderSource();
    const char* fragment_src = getFragmentShaderSource();
    
    m_shader_program = createShaderProgram(vertex_src, fragment_src);
    
    if (m_shader_program == 0) {
        std::cerr << "Failed to create shader program for test cube!" << std::endl;
    } else {
        std::cout << "Test cube shader program created successfully" << std::endl;
    }
}

void TestCubeRenderer::setupBuffers() {
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);
    
    glBindVertexArray(m_vao);
    
    // Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(float), m_vertices.data(), GL_STATIC_DRAW);
    
    // Element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

const char* TestCubeRenderer::getVertexShaderSource() {
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

const char* TestCubeRenderer::getFragmentShaderSource() {
    return R"(
#version 330 core
out vec4 FragColor;

void main() {
    // Bright red color for the test cube
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
)";
}

GLuint TestCubeRenderer::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check compilation
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Test cube shader compilation failed: " << info_log << std::endl;
        return 0;
    }
    
    return shader;
}

GLuint TestCubeRenderer::createShaderProgram(const char* vertex_source, const char* fragment_source) {
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
        std::cerr << "Test cube shader program linking failed: " << info_log << std::endl;
        return 0;
    }
    
    // Clean up individual shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    return program;
} 