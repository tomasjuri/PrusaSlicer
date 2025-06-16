#pragma once
#include "GCodeInputData.hpp"
#include "Types.hpp"

namespace libvgcode {

class Viewer {
public:
    Viewer() = default;
    ~Viewer() = default;
    
    // Initialize the viewer
    bool init(int width, int height) {
        m_width = width;
        m_height = height;
        m_initialized = true;
        return true;
    }
    
    // Load G-code data
    bool load(const GCodeInputData& data) {
        m_data = data;
        m_loaded = true;
        return true;
    }
    
    // Render the G-code visualization
    void render(const Mat4x4& view_matrix, const Mat4x4& projection_matrix) {
        if (!m_initialized || !m_loaded) return;
        
        // Simplified rendering - in a real implementation this would use OpenGL
        // to render the toolpaths stored in m_data
    }
    
    // Clear loaded data
    void clear() {
        m_data.clear();
        m_loaded = false;
    }
    
    // Check if data is loaded
    bool is_loaded() const { return m_loaded; }
    
    // Get loaded data
    const GCodeInputData& get_data() const { return m_data; }
    
private:
    GCodeInputData m_data;
    int m_width = 0;
    int m_height = 0;
    bool m_initialized = false;
    bool m_loaded = false;
};

} // namespace libvgcode 