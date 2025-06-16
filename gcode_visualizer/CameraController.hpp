#pragma once

#include <array>

// Forward declare our simple matrix type
struct Mat4x4;

class CameraController {
public:
    CameraController();
    ~CameraController() = default;
    
    // Set up camera for top view, looking down at the build plate
    void setTopView(int width, int height);
    
    // Set camera position and target manually
    void setPosition(const std::array<float, 3>& position, const std::array<float, 3>& target);
    
    // Get view matrix for rendering
    Mat4x4 getViewMatrix() const;
    
    // Get projection matrix for rendering
    Mat4x4 getProjectionMatrix() const;
    
    // Camera parameters
    void setFieldOfView(float fov) { m_fov = fov; }
    void setNearPlane(float near_plane) { m_near_plane = near_plane; }
    void setFarPlane(float far_plane) { m_far_plane = far_plane; }
    
private:
    std::array<float, 3> m_position;
    std::array<float, 3> m_target;
    std::array<float, 3> m_up;
    
    float m_fov;        // Field of view in degrees
    float m_near_plane;
    float m_far_plane;
    float m_aspect_ratio;
    
    // Helper methods for matrix calculations
    Mat4x4 createLookAtMatrix(
        const std::array<float, 3>& eye,
        const std::array<float, 3>& center,
        const std::array<float, 3>& up) const;
    
    Mat4x4 createPerspectiveMatrix(
        float fov, float aspect, float near_plane, float far_plane) const;
}; 