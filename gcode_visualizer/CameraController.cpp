#include "CameraController.hpp"
#include "GCodeVisualizerApp.hpp"  // For Mat4x4 definition
#include <cmath>
#include <iostream>
#include <algorithm>

CameraController::CameraController() 
    : m_position({0.0f, 0.0f, 500.0f})      // 50cm above center of bed (now at origin)
    , m_target({0.0f, 0.0f, 0.0f})          // Center of bed at origin
    , m_up({0.0f, 1.0f, 0.0f})              // Y-up
    , m_fov(45.0f)
    , m_near_plane(1.0f)
    , m_far_plane(1000.0f)
    , m_aspect_ratio(1.0f)
{
    std::cout << "Camera initialized 50cm above bed center" << std::endl;
}

void CameraController::setTopView(int width, int height) {
    m_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    // Position camera 50cm above the center of the print bed (now at origin)
    m_position = {0.0f, 0.0f, 500.0f};      // Center of bed at origin + 50cm height
    m_target = {0.0f, 0.0f, 0.0f};          // Center of bed at origin
    m_up = {0.0f, 1.0f, 0.0f};              // Y is up
    
    std::cout << "Camera set to top view: 50cm above bed" << std::endl;
}

void CameraController::setPosition(const std::array<float, 3>& position, const std::array<float, 3>& target) {
    m_position = position;
    m_target = target;
    
    float distance = getDistanceFromTarget();
    std::cout << "Camera position set to (" << position[0] << ", " << position[1] << ", " << position[2] 
              << ") looking at (" << target[0] << ", " << target[1] << ", " << target[2] 
              << ") distance: " << distance << "mm" << std::endl;
}

void CameraController::setDistanceFromSurface(float distance_mm) {
    // Keep current target, but adjust position to be distance_mm above it
    m_position[0] = m_target[0];
    m_position[1] = m_target[1];
    m_position[2] = m_target[2] + distance_mm;
    
    std::cout << "Camera distance set to " << distance_mm << "mm from surface" << std::endl;
}

void CameraController::setAngleView(float angle_degrees, float distance_mm, int width, int height) {
    m_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    // Calculate position at an angle
    float angle_rad = angle_degrees * M_PI / 180.0f;
    
    // Position camera at an angle, looking down at the print
    float x_offset = distance_mm * std::sin(angle_rad) * 0.8f;  // 0.8 to move slightly to side
    float z_height = distance_mm * std::cos(angle_rad);
    
    m_position = {
        m_target[0] + x_offset,
        m_target[1] - distance_mm * 0.3f,  // Move back slightly
        m_target[2] + z_height
    };
    
    std::cout << "Camera set to angled view: " << angle_degrees << "° at " << distance_mm << "mm distance" << std::endl;
}

void CameraController::setOptimalView(float print_min_x, float print_max_x, 
                                     float print_min_y, float print_max_y, 
                                     float print_max_z, int width, int height) {
    m_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    // Calculate print center
    float center_x = (print_min_x + print_max_x) / 2.0f;
    float center_y = (print_min_y + print_max_y) / 2.0f;
    float center_z = print_max_z / 2.0f;
    
    m_target = {center_x, center_y, center_z};
    
    // Calculate optimal distance based on print size
    float print_width = print_max_x - print_min_x;
    float print_height = print_max_y - print_min_y;
    float print_depth = print_max_z;
    
    calculateOptimalDistance(print_width, print_height, print_depth);
    
    std::cout << "Camera set to optimal view for print bounds: " 
              << print_width << "x" << print_height << "x" << print_depth << "mm" << std::endl;
}

float CameraController::getDistanceFromTarget() const {
    float dx = m_position[0] - m_target[0];
    float dy = m_position[1] - m_target[1];
    float dz = m_position[2] - m_target[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void CameraController::calculateOptimalDistance(float print_width, float print_height, float print_depth) {
    // Calculate distance needed to fit the print in view
    float max_dimension = std::max({print_width, print_height, print_depth});
    
    // Use field of view to calculate required distance
    float fov_rad = m_fov * M_PI / 180.0f;
    float distance = (max_dimension / 2.0f) / std::tan(fov_rad / 2.0f);
    
    // Add some margin for better view
    distance *= 1.5f;
    
    // Minimum distance of 500mm (50cm) as requested
    distance = std::max(distance, 500.0f);
    
    // Position camera at an angle for better 3D view
    float angle = 30.0f * M_PI / 180.0f;  // 30 degree angle
    
    m_position = {
        m_target[0] + distance * std::sin(angle),
        m_target[1] - distance * 0.5f,
        m_target[2] + distance * std::cos(angle)
    };
    
    std::cout << "Calculated optimal distance: " << distance << "mm" << std::endl;
}

Mat4x4 CameraController::getViewMatrix() const {
    return createLookAtMatrix(m_position, m_target, m_up);
}

Mat4x4 CameraController::getProjectionMatrix() const {
    return createPerspectiveMatrix(m_fov, m_aspect_ratio, m_near_plane, m_far_plane);
}

Mat4x4 CameraController::createLookAtMatrix(
    const std::array<float, 3>& eye,
    const std::array<float, 3>& center,
    const std::array<float, 3>& up) const {
    
    // Calculate camera coordinate system
    std::array<float, 3> f = {
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2]
    };
    
    // Normalize f
    float f_len = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0] /= f_len; f[1] /= f_len; f[2] /= f_len;
    
    // Calculate right vector (f cross up)
    std::array<float, 3> s = {
        f[1]*up[2] - f[2]*up[1],
        f[2]*up[0] - f[0]*up[2],
        f[0]*up[1] - f[1]*up[0]
    };
    
    // Normalize s
    float s_len = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    s[0] /= s_len; s[1] /= s_len; s[2] /= s_len;
    
    // Calculate up vector (s cross f)
    std::array<float, 3> u = {
        s[1]*f[2] - s[2]*f[1],
        s[2]*f[0] - s[0]*f[2],
        s[0]*f[1] - s[1]*f[0]
    };
    
    Mat4x4 result;
    
    // First row
    result.m_data[0] = s[0];
    result.m_data[1] = u[0];
    result.m_data[2] = -f[0];
    result.m_data[3] = 0.0f;
    
    // Second row
    result.m_data[4] = s[1];
    result.m_data[5] = u[1];
    result.m_data[6] = -f[1];
    result.m_data[7] = 0.0f;
    
    // Third row
    result.m_data[8] = s[2];
    result.m_data[9] = u[2];
    result.m_data[10] = -f[2];
    result.m_data[11] = 0.0f;
    
    // Fourth row (translation)
    result.m_data[12] = -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]);
    result.m_data[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    result.m_data[14] = f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2];
    result.m_data[15] = 1.0f;
    
    return result;
}

Mat4x4 CameraController::createPerspectiveMatrix(
    float fov, float aspect, float near_plane, float far_plane) const {
    
    float fov_rad = fov * M_PI / 180.0f;
    float tan_half_fov = std::tan(fov_rad / 2.0f);
    
    Mat4x4 result;
    
    result.m_data[0] = 1.0f / (aspect * tan_half_fov);
    result.m_data[1] = 0.0f;
    result.m_data[2] = 0.0f;
    result.m_data[3] = 0.0f;
    
    result.m_data[4] = 0.0f;
    result.m_data[5] = 1.0f / tan_half_fov;
    result.m_data[6] = 0.0f;
    result.m_data[7] = 0.0f;
    
    result.m_data[8] = 0.0f;
    result.m_data[9] = 0.0f;
    result.m_data[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    result.m_data[11] = -1.0f;
    
    result.m_data[12] = 0.0f;
    result.m_data[13] = 0.0f;
    result.m_data[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    result.m_data[15] = 0.0f;
    
    return result;
} 