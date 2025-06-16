#include "CameraController.hpp"
#include "GCodeVisualizerApp.hpp"  // For Mat4x4 definition
#include <cmath>

CameraController::CameraController() 
    : m_position({125.0f, 105.0f, 100.0f})  // 10cm above center of Prusa bed
    , m_target({125.0f, 105.0f, 0.0f})      // Center of bed
    , m_up({0.0f, 1.0f, 0.0f})              // Y-up
    , m_fov(45.0f)
    , m_near_plane(1.0f)
    , m_far_plane(1000.0f)
    , m_aspect_ratio(1.0f)
{
}

void CameraController::setTopView(int width, int height) {
    m_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    // Position camera 10cm above the center of the print bed
    m_position = {125.0f, 105.0f, 100.0f};  // Center of standard Prusa bed + 10cm height
    m_target = {125.0f, 105.0f, 0.0f};      // Center of bed at Z=0
    m_up = {0.0f, 1.0f, 0.0f};              // Y is up
}

void CameraController::setPosition(const std::array<float, 3>& position, const std::array<float, 3>& target) {
    m_position = position;
    m_target = target;
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