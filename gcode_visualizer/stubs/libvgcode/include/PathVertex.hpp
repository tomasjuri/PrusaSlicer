#pragma once
#include "Types.hpp"
#include <array>

namespace libvgcode {

struct PathVertex {
    std::array<float, 3> position = {0.0f, 0.0f, 0.0f};
    float extrusion = 0.0f;
    float speed = 0.0f;
    float layer_time = 0.0f;
    float elapsed_time = 0.0f;
    float layer_height = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float volumetric_rate = 0.0f;
    float temperature = 0.0f;
    float fan_speed = 0.0f;
    
    EMoveType type = EMoveType::Noop;
    EGCodeExtrusionRole role = EGCodeExtrusionRole::None;
    
    int tool_id = 0;
    int color_id = 0;
    int layer_id = 0;
    
    PathVertex() = default;
    
    PathVertex(const std::array<float, 3>& pos, EMoveType move_type = EMoveType::Noop) 
        : position(pos), type(move_type) {}
};

} // namespace libvgcode 