#include "GCodeConverter.hpp"
#include <iostream>

GCodeConverter::GCodeConverter() {
}

libvgcode::GCodeInputData GCodeConverter::convert(const Slic3r::GCodeProcessorResult& result) {
    libvgcode::GCodeInputData data;
    
    // Set spiral vase mode
    data.spiral_vase_mode = result.spiral_vase_mode;
    
    // Create default color palettes
    data.tools_colors = createDefaultToolColors();
    data.color_print_colors = createDefaultColorPrintColors();
    
    // Reserve space for vertices
    data.vertices.reserve(result.moves.size());
    
    std::cout << "Converting " << result.moves.size() << " moves to libvgcode format..." << std::endl;
    
    // Convert each move to a PathVertex
    for (const auto& move : result.moves) {
        auto vertex = convertMoveVertex(move);
        data.vertices.push_back(vertex);
    }
    
    std::cout << "Conversion complete: " << data.vertices.size() << " vertices created" << std::endl;
    
    return data;
}

libvgcode::PathVertex GCodeConverter::convertMoveVertex(const Slic3r::GCodeProcessorResult::MoveVertex& move) {
    libvgcode::PathVertex vertex;
    
    // Position (using array indexing instead of .x(), .y(), .z())
    vertex.position[0] = move.position[0];
    vertex.position[1] = move.position[1]; 
    vertex.position[2] = move.position[2];
    
    // Basic properties that exist in our stubs
    vertex.layer_height = move.layer_height;
    vertex.width = move.width;
    vertex.height = move.height;
    vertex.speed = move.speed;
    vertex.extrusion = move.extrusion;
    vertex.elapsed_time = move.elapsed_time;
    vertex.layer_time = move.layer_time;
    vertex.volumetric_rate = move.volumetric_rate;
    // These fields don't exist in our stub, so set defaults
    vertex.temperature = 200.0f;  // Default temperature  
    vertex.fan_speed = 0.0f;      // Default fan speed
    
    // Move classification
    vertex.role = convertExtrusionRole(move.role);
    vertex.type = convertMoveType(move.type);
    
    // IDs and metadata
    vertex.layer_id = move.layer_id;
    vertex.tool_id = move.tool_id;
    vertex.color_id = move.color_id;
    
    return vertex;
}

libvgcode::EMoveType GCodeConverter::convertMoveType(Slic3r::EMoveType type) {
    switch (type) {
        case Slic3r::EMoveType::Noop: return libvgcode::EMoveType::Noop;
        case Slic3r::EMoveType::Retract: return libvgcode::EMoveType::Retract;
        case Slic3r::EMoveType::Unretract: return libvgcode::EMoveType::Unretract;
        case Slic3r::EMoveType::Seam: return libvgcode::EMoveType::Seam;
        case Slic3r::EMoveType::ToolChange: return libvgcode::EMoveType::ToolChange;
        case Slic3r::EMoveType::ColorChange: return libvgcode::EMoveType::ColorChange;
        case Slic3r::EMoveType::PausePrint: return libvgcode::EMoveType::PausePrint;
        case Slic3r::EMoveType::CustomGCode: return libvgcode::EMoveType::CustomGCode;
        case Slic3r::EMoveType::Travel: return libvgcode::EMoveType::Travel;
        case Slic3r::EMoveType::Wipe: return libvgcode::EMoveType::Wipe;
        case Slic3r::EMoveType::Extrude: return libvgcode::EMoveType::Extrude;
        default: return libvgcode::EMoveType::Noop;
    }
}

libvgcode::EGCodeExtrusionRole GCodeConverter::convertExtrusionRole(Slic3r::GCodeExtrusionRole role) {
    switch (role) {
        case Slic3r::GCodeExtrusionRole::None: return libvgcode::EGCodeExtrusionRole::None;
        case Slic3r::GCodeExtrusionRole::Perimeter: return libvgcode::EGCodeExtrusionRole::Perimeter;
        case Slic3r::GCodeExtrusionRole::ExternalPerimeter: return libvgcode::EGCodeExtrusionRole::ExternalPerimeter;
        case Slic3r::GCodeExtrusionRole::OverhangPerimeter: return libvgcode::EGCodeExtrusionRole::OverhangPerimeter;
        case Slic3r::GCodeExtrusionRole::InternalInfill: return libvgcode::EGCodeExtrusionRole::InternalInfill;
        case Slic3r::GCodeExtrusionRole::SolidInfill: return libvgcode::EGCodeExtrusionRole::SolidInfill;
        case Slic3r::GCodeExtrusionRole::TopSolidInfill: return libvgcode::EGCodeExtrusionRole::TopSolidInfill;
        case Slic3r::GCodeExtrusionRole::Ironing: return libvgcode::EGCodeExtrusionRole::Ironing;
        case Slic3r::GCodeExtrusionRole::BridgeInfill: return libvgcode::EGCodeExtrusionRole::BridgeInfill;
        case Slic3r::GCodeExtrusionRole::GapFill: return libvgcode::EGCodeExtrusionRole::GapFill;
        case Slic3r::GCodeExtrusionRole::Skirt: return libvgcode::EGCodeExtrusionRole::Skirt;
        case Slic3r::GCodeExtrusionRole::SupportMaterial: return libvgcode::EGCodeExtrusionRole::SupportMaterial;
        case Slic3r::GCodeExtrusionRole::SupportMaterialInterface: return libvgcode::EGCodeExtrusionRole::SupportMaterialInterface;
        case Slic3r::GCodeExtrusionRole::WipeTower: return libvgcode::EGCodeExtrusionRole::WipeTower;
        case Slic3r::GCodeExtrusionRole::Custom: return libvgcode::EGCodeExtrusionRole::Custom;
        default: return libvgcode::EGCodeExtrusionRole::None;
    }
}

std::vector<std::array<float, 4>> GCodeConverter::createDefaultToolColors() {
    std::vector<std::array<float, 4>> palette;
    
    // Standard extruder colors (RGBA format)
    palette.push_back({1.0f, 0.0f, 0.0f, 1.0f});  // Red
    palette.push_back({0.0f, 1.0f, 0.0f, 1.0f});  // Green
    palette.push_back({0.0f, 0.0f, 1.0f, 1.0f});  // Blue
    palette.push_back({1.0f, 1.0f, 0.0f, 1.0f});  // Yellow
    palette.push_back({1.0f, 0.0f, 1.0f, 1.0f});  // Magenta
    palette.push_back({0.0f, 1.0f, 1.0f, 1.0f});  // Cyan
    palette.push_back({1.0f, 0.5f, 0.0f, 1.0f});  // Orange
    palette.push_back({0.5f, 0.0f, 1.0f, 1.0f});  // Purple
    
    return palette;
}

std::vector<std::array<float, 4>> GCodeConverter::createDefaultColorPrintColors() {
    std::vector<std::array<float, 4>> palette;
    
    // Color print palette (similar to tools but different hues)
    palette.push_back({0.8f, 0.2f, 0.2f, 1.0f}); // Dark red
    palette.push_back({0.2f, 0.8f, 0.2f, 1.0f}); // Dark green
    palette.push_back({0.2f, 0.2f, 0.8f, 1.0f}); // Dark blue
    palette.push_back({0.8f, 0.8f, 0.2f, 1.0f}); // Dark yellow
    palette.push_back({0.8f, 0.2f, 0.8f, 1.0f}); // Dark magenta
    palette.push_back({0.2f, 0.8f, 0.8f, 1.0f}); // Dark cyan
    palette.push_back({0.8f, 0.4f, 0.2f, 1.0f}); // Dark orange
    palette.push_back({0.4f, 0.2f, 0.8f, 1.0f}); // Dark purple
    
    return palette;
} 