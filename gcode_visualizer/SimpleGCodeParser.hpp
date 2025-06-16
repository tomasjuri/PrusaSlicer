#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <cmath>
#include <optional>
#include <tuple>

namespace SimpleGCode {

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

enum class MoveType {
    Travel,
    Extrusion,
    Retraction,
    Unretraction
};

struct GCodeMove {
    Vec3 start_pos;
    Vec3 end_pos;
    float extrusion_delta = 0.0f;  // Amount of filament extruded (can be negative for retraction)
    float feedrate = 0.0f;         // mm/min
    MoveType type = MoveType::Travel;
    int layer_id = 0;
    float width = 0.4f;            // Extrusion width in mm
    float height = 0.2f;           // Layer height in mm
    std::string original_line;     // Original G-code line for debugging
};

struct GCodeStats {
    Vec3 min_pos = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 max_pos = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    int total_layers = 0;
    size_t total_moves = 0;
    size_t extrusion_moves = 0;
    size_t travel_moves = 0;
    size_t retraction_moves = 0;
    float total_filament_used = 0.0f;  // mm of filament
    float estimated_time = 0.0f;       // minutes
    std::string filename;
};

class GCodeParser {
public:
    GCodeParser();
    
    // Parse a G-code file and extract move data
    bool parseFile(const std::string& filename);
    
    // Get parsed moves
    const std::vector<GCodeMove>& getMoves() const { return m_moves; }
    
    // Get statistics
    const GCodeStats& getStats() const { return m_stats; }
    
    // Print detailed analysis
    void printAnalysis() const;
    
    // Get print bounds (min_x, max_x, min_y, max_y, max_z)
    std::optional<std::tuple<float, float, float, float, float>> getPrintBounds() const;
    
private:
    std::vector<GCodeMove> m_moves;
    GCodeStats m_stats;
    
    // Current parser state
    Vec3 m_current_pos = Vec3(0, 0, 0);
    float m_current_e = 0.0f;
    float m_current_feedrate = 1500.0f;  // Default feedrate
    int m_current_layer = 0;
    bool m_absolute_positioning = true;
    bool m_absolute_extrusion = true;
    
    // Parse a single G-code line
    bool parseLine(const std::string& line, int line_number);
    
    // Extract parameter value from G-code line (e.g., "X123.45" -> 123.45)
    bool extractFloat(const std::string& line, char param, float& value) const;
    
    // Handle G0/G1 movement commands
    void handleMovement(const std::string& line, bool is_g0);
    
    // Handle layer change detection
    void detectLayerChange(const std::string& line);
    
    // Update statistics
    void updateStats(const GCodeMove& move);
    
    // Helper to determine move type
    MoveType determineMoveType(float e_delta, const Vec3& position_delta) const;
    
    // Helper to clean up G-code line (remove comments, trim whitespace)
    std::string cleanLine(const std::string& line) const;
}; 

}  // namespace SimpleGCode 