#pragma once
#include <vector>
#include <string>
#include <array>

namespace Slic3r {

// Enums for move types
enum class EMoveType {
    Noop = 0,
    Retract,
    Unretract,
    Seam,
    ToolChange,
    ColorChange,
    PausePrint,
    CustomGCode,
    Travel,
    Wipe,
    Extrude
};

enum class GCodeExtrusionRole {
    None = 0,
    Perimeter,
    ExternalPerimeter,
    OverhangPerimeter,
    InternalInfill,
    SolidInfill,
    TopSolidInfill,
    Ironing,
    BridgeInfill,
    GapFill,
    Skirt,
    SupportMaterial,
    SupportMaterialInterface,
    WipeTower,
    Custom
};

struct GCodeProcessorResult {
    struct MoveVertex {
        std::array<float, 3> position = {0.0f, 0.0f, 0.0f};
        float extrusion = 0.0f;
        float speed = 0.0f;
        float layer_time = 0.0f;
        float elapsed_time = 0.0f;
        float layer_height = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float volumetric_rate = 0.0f;
        
        EMoveType type = EMoveType::Noop;
        GCodeExtrusionRole role = GCodeExtrusionRole::None;
        
        int tool_id = 0;
        int color_id = 0;
        int layer_id = 0;
    };
    
    std::vector<MoveVertex> moves;
    bool spiral_vase_mode = false;
    float estimated_print_time = 0.0f;
    float filament_used = 0.0f;
    
    void clear() {
        moves.clear();
        spiral_vase_mode = false;
        estimated_print_time = 0.0f;
        filament_used = 0.0f;
    }
};

class GCodeProcessor {
public:
    GCodeProcessor() = default;
    ~GCodeProcessor() = default;
    
    // Process a G-code file and return results
    GCodeProcessorResult process_file(const std::string& filename) {
        // Simplified G-code parsing
        GCodeProcessorResult result;
        
        // This is a very basic stub - real implementation would parse G-code
        // For now, create some dummy data to demonstrate the concept
        
        // Add some sample moves to show the concept works
        GCodeProcessorResult::MoveVertex vertex;
        vertex.position = {125.0f, 105.0f, 0.2f};  // Center of Prusa bed
        vertex.type = EMoveType::Travel;
        result.moves.push_back(vertex);
        
        vertex.position = {100.0f, 100.0f, 0.2f};
        vertex.type = EMoveType::Extrude;
        vertex.role = GCodeExtrusionRole::Perimeter;
        result.moves.push_back(vertex);
        
        vertex.position = {150.0f, 100.0f, 0.2f};
        result.moves.push_back(vertex);
        
        vertex.position = {150.0f, 110.0f, 0.2f};
        result.moves.push_back(vertex);
        
        vertex.position = {100.0f, 110.0f, 0.2f};
        result.moves.push_back(vertex);
        
        return result;
    }
    
    // Process G-code from a string
    GCodeProcessorResult process_string(const std::string& gcode) {
        // Stub implementation
        return process_file("");
    }
};

} // namespace Slic3r 