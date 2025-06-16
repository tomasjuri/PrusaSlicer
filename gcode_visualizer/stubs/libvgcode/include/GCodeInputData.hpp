#pragma once
#include "PathVertex.hpp"
#include <vector>
#include <array>

namespace libvgcode {

struct GCodeInputData {
    // Main path data
    std::vector<PathVertex> vertices;
    
    // Configuration
    bool spiral_vase_mode = false;
    
    // Color palettes
    std::vector<std::array<float, 4>> tools_colors;  // RGBA colors for each tool
    std::vector<std::array<float, 4>> color_print_colors;  // RGBA colors for color printing
    
    // Statistics (optional)
    float estimated_print_time = 0.0f;
    float filament_used = 0.0f;
    
    GCodeInputData() = default;
    
    void clear() {
        vertices.clear();
        tools_colors.clear();
        color_print_colors.clear();
        spiral_vase_mode = false;
        estimated_print_time = 0.0f;
        filament_used = 0.0f;
    }
    
    bool empty() const {
        return vertices.empty();
    }
    
    size_t size() const {
        return vertices.size();
    }
};

} // namespace libvgcode 