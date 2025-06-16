#include "SimpleGCodeParser.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace SimpleGCode {

GCodeParser::GCodeParser() {
}

bool GCodeParser::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open G-code file: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Parsing G-code file: " << filename << std::endl;
    
    // Initialize stats
    m_stats = GCodeStats();
    m_stats.filename = filename;
    m_moves.clear();
    
    // Reset parser state
    m_current_pos = Vec3(0, 0, 0);
    m_current_e = 0.0f;
    m_current_feedrate = 1500.0f;
    m_current_layer = 0;
    m_absolute_positioning = true;
    m_absolute_extrusion = true;
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        if (!parseLine(line, line_number)) {
            // Continue parsing even if individual lines fail
        }
        
        // Progress indicator for large files
        if (line_number % 10000 == 0) {
            std::cout << "Processed " << line_number << " lines, " << m_moves.size() << " moves..." << std::endl;
        }
    }
    
    file.close();
    
    // Finalize stats
    m_stats.total_moves = m_moves.size();
    
    std::cout << "Parsing complete: " << m_stats.total_moves << " moves, " 
              << m_stats.total_layers << " layers" << std::endl;
    
    return true;
}

void GCodeParser::printAnalysis() const {
    std::cout << "\n=== G-Code Analysis ===" << std::endl;
    std::cout << "File: " << m_stats.filename << std::endl;
    std::cout << "Total moves: " << m_stats.total_moves << std::endl;
    std::cout << "Layers: " << m_stats.total_layers << std::endl;
    
    std::cout << "\nMove breakdown:" << std::endl;
    std::cout << "  Extrusion moves: " << m_stats.extrusion_moves << std::endl;
    std::cout << "  Travel moves: " << m_stats.travel_moves << std::endl;
    std::cout << "  Retraction moves: " << m_stats.retraction_moves << std::endl;
    
    std::cout << "\nPrint bounds:" << std::endl;
    std::cout << "  X: " << m_stats.min_pos.x << " to " << m_stats.max_pos.x 
              << " mm (size: " << (m_stats.max_pos.x - m_stats.min_pos.x) << " mm)" << std::endl;
    std::cout << "  Y: " << m_stats.min_pos.y << " to " << m_stats.max_pos.y 
              << " mm (size: " << (m_stats.max_pos.y - m_stats.min_pos.y) << " mm)" << std::endl;
    std::cout << "  Z: " << m_stats.min_pos.z << " to " << m_stats.max_pos.z 
              << " mm (height: " << (m_stats.max_pos.z - m_stats.min_pos.z) << " mm)" << std::endl;
    
    std::cout << "\nFilament usage:" << std::endl;
    std::cout << "  Total filament: " << m_stats.total_filament_used << " mm" << std::endl;
    
    if (m_stats.estimated_time > 0) {
        std::cout << "  Estimated time: " << m_stats.estimated_time << " minutes" << std::endl;
    }
    
    // Show some example moves for debugging
    if (!m_moves.empty()) {
        std::cout << "\nFirst few moves (for debugging):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), m_moves.size()); i++) {
            const auto& move = m_moves[i];
            std::cout << "  Move " << i << ": " 
                      << "(" << move.start_pos.x << "," << move.start_pos.y << "," << move.start_pos.z << ") -> "
                      << "(" << move.end_pos.x << "," << move.end_pos.y << "," << move.end_pos.z << ") "
                      << "E:" << move.extrusion_delta << " "
                      << "Type:" << (int)move.type << " "
                      << "Layer:" << move.layer_id << std::endl;
        }
    }
    
    std::cout << "======================\n" << std::endl;
}

bool GCodeParser::parseLine(const std::string& line, int line_number) {
    std::string clean = cleanLine(line);
    if (clean.empty()) {
        return true;  // Empty line is ok
    }
    
    // Convert to uppercase for easier parsing
    std::transform(clean.begin(), clean.end(), clean.begin(), ::toupper);
    
    // Look for layer change indicators
    detectLayerChange(line);
    
    // Handle G-code commands
    if (clean.find("G0 ") == 0 || clean.find("G0\t") == 0) {
        handleMovement(clean, true);
    } else if (clean.find("G1 ") == 0 || clean.find("G1\t") == 0) {
        handleMovement(clean, false);
    } else if (clean.find("G90") == 0) {
        m_absolute_positioning = true;
    } else if (clean.find("G91") == 0) {
        m_absolute_positioning = false;
    } else if (clean.find("M82") == 0) {
        m_absolute_extrusion = true;
    } else if (clean.find("M83") == 0) {
        m_absolute_extrusion = false;
    }
    
    return true;
}

bool GCodeParser::extractFloat(const std::string& line, char param, float& value) const {
    size_t pos = line.find(param);
    if (pos == std::string::npos) {
        return false;
    }
    
    pos++;  // Skip the parameter letter
    
    // Find the end of the number
    size_t end_pos = pos;
    while (end_pos < line.length()) {
        char c = line[end_pos];
        if (c == ' ' || c == '\t' || c == ';') {
            break;
        }
        end_pos++;
    }
    
    try {
        std::string number_str = line.substr(pos, end_pos - pos);
        value = std::stof(number_str);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void GCodeParser::handleMovement(const std::string& line, bool is_g0) {
    Vec3 target_pos = m_current_pos;
    float target_e = m_current_e;
    float feedrate = m_current_feedrate;
    
    // Extract parameters
    float x, y, z, e, f;
    
    if (extractFloat(line, 'X', x)) {
        target_pos.x = m_absolute_positioning ? x : m_current_pos.x + x;
    }
    if (extractFloat(line, 'Y', y)) {
        target_pos.y = m_absolute_positioning ? y : m_current_pos.y + y;
    }
    if (extractFloat(line, 'Z', z)) {
        target_pos.z = m_absolute_positioning ? z : m_current_pos.z + z;
    }
    if (extractFloat(line, 'E', e)) {
        target_e = m_absolute_extrusion ? e : m_current_e + e;
    }
    if (extractFloat(line, 'F', f)) {
        feedrate = f;
        m_current_feedrate = f;
    }
    
    // Create move
    GCodeMove move;
    move.start_pos = m_current_pos;
    move.end_pos = target_pos;
    move.extrusion_delta = target_e - m_current_e;
    move.feedrate = feedrate;
    move.layer_id = m_current_layer;
    move.original_line = line;
    
    // Determine move type
    Vec3 position_delta = target_pos - m_current_pos;
    move.type = determineMoveType(move.extrusion_delta, position_delta);
    
    // Only add moves that actually move somewhere
    if (position_delta.length() > 0.001f || std::abs(move.extrusion_delta) > 0.001f) {
        updateStats(move);
        m_moves.push_back(move);
    }
    
    // Update current state
    m_current_pos = target_pos;
    m_current_e = target_e;
}

void GCodeParser::detectLayerChange(const std::string& line) {
    // Look for common layer change indicators
    if (line.find(";LAYER:") != std::string::npos ||
        line.find("; layer ") != std::string::npos ||
        line.find(";LAYER ") != std::string::npos ||
        line.find("; LAYER:") != std::string::npos) {
        
        // Try to extract layer number
        size_t pos = line.find_last_of("0123456789");
        if (pos != std::string::npos) {
            size_t start = pos;
            while (start > 0 && std::isdigit(line[start - 1])) {
                start--;
            }
            try {
                int layer = std::stoi(line.substr(start, pos - start + 1));
                m_current_layer = layer;
                m_stats.total_layers = std::max(m_stats.total_layers, layer + 1);
            } catch (const std::exception&) {
                // Failed to parse layer number, use Z height as layer indicator
                if (m_current_pos.z > m_stats.total_layers * 0.2f) {
                    m_current_layer = (int)(m_current_pos.z / 0.2f);
                    m_stats.total_layers = m_current_layer + 1;
                }
            }
        }
    }
}

void GCodeParser::updateStats(const GCodeMove& move) {
    // Update bounds
    m_stats.min_pos.x = std::min(m_stats.min_pos.x, std::min(move.start_pos.x, move.end_pos.x));
    m_stats.min_pos.y = std::min(m_stats.min_pos.y, std::min(move.start_pos.y, move.end_pos.y));
    m_stats.min_pos.z = std::min(m_stats.min_pos.z, std::min(move.start_pos.z, move.end_pos.z));
    
    m_stats.max_pos.x = std::max(m_stats.max_pos.x, std::max(move.start_pos.x, move.end_pos.x));
    m_stats.max_pos.y = std::max(m_stats.max_pos.y, std::max(move.start_pos.y, move.end_pos.y));
    m_stats.max_pos.z = std::max(m_stats.max_pos.z, std::max(move.start_pos.z, move.end_pos.z));
    
    // Update move counts and filament usage
    switch (move.type) {
        case MoveType::Extrusion:
            m_stats.extrusion_moves++;
            if (move.extrusion_delta > 0) {
                m_stats.total_filament_used += move.extrusion_delta;
            }
            break;
        case MoveType::Travel:
            m_stats.travel_moves++;
            break;
        case MoveType::Retraction:
        case MoveType::Unretraction:
            m_stats.retraction_moves++;
            break;
    }
    
    // Estimate time (very rough)
    Vec3 distance = move.end_pos - move.start_pos;
    if (move.feedrate > 0) {
        m_stats.estimated_time += distance.length() / move.feedrate;  // minutes
    }
}

MoveType GCodeParser::determineMoveType(float e_delta, const Vec3& position_delta) const {
    const float MIN_MOVEMENT = 0.001f;
    const float MIN_EXTRUSION = 0.001f;
    
    bool has_movement = position_delta.length() > MIN_MOVEMENT;
    bool has_extrusion = std::abs(e_delta) > MIN_EXTRUSION;
    
    if (has_extrusion && e_delta < 0 && !has_movement) {
        return MoveType::Retraction;
    } else if (has_extrusion && e_delta > 0 && !has_movement) {
        return MoveType::Unretraction;
    } else if (has_extrusion && e_delta > 0 && has_movement) {
        return MoveType::Extrusion;
    } else {
        return MoveType::Travel;
    }
}

std::string GCodeParser::cleanLine(const std::string& line) const {
    std::string clean = line;
    
    // Remove comments (everything after ';')
    size_t comment_pos = clean.find(';');
    if (comment_pos != std::string::npos) {
        clean = clean.substr(0, comment_pos);
    }
    
    // Trim whitespace
    clean.erase(0, clean.find_first_not_of(" \t\r\n"));
    clean.erase(clean.find_last_not_of(" \t\r\n") + 1);
    
    return clean;
}

} // namespace SimpleGCode 