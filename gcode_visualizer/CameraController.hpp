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
    
    // Set camera distance from print surface (in mm)
    void setDistanceFromSurface(float distance_mm);
    
    // Set camera angle and distance for better visualization
    void setAngleView(float angle_degrees, float distance_mm, int width, int height);
    
    // Move camera to optimal position based on print bounds
    void setOptimalView(float print_min_x, float print_max_x, 
                       float print_min_y, float print_max_y, 
                       float print_max_z, int width, int height);
    
    // Get view matrix for rendering
    Mat4x4 getViewMatrix() const;
    
    // Get projection matrix for rendering
    Mat4x4 getProjectionMatrix() const;
    
    // Camera parameters
    void setFieldOfView(float fov) { m_fov = fov; }
    void setNearPlane(float near_plane) { m_near_plane = near_plane; }
    void setFarPlane(float far_plane) { m_far_plane = far_plane; }
    
    // Get current camera info
    std::array<float, 3> getPosition() const { return m_position; }
    std::array<float, 3> getTarget() const { return m_target; }
    float getDistanceFromTarget() const;
    
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
    
    // Helper methods for positioning
    void calculateOptimalDistance(float print_width, float print_height, float print_depth);
}; 