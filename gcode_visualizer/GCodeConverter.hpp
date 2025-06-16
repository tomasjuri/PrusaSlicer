#pragma once

#include "stubs/libvgcode/include/GCodeInputData.hpp"
#include "stubs/libslic3r/GCode/GCodeProcessor.hpp"

// Converter class to transform PrusaSlicer's GCodeProcessor results
// into libvgcode::GCodeInputData format for visualization
class GCodeConverter {
public:
    GCodeConverter();
    ~GCodeConverter() = default;
    
    // Convert GCodeProcessor result to libvgcode format
    libvgcode::GCodeInputData convert(const Slic3r::GCodeProcessorResult& result);
    
private:
    // Helper methods for conversion
    libvgcode::PathVertex convertMoveVertex(const Slic3r::GCodeProcessorResult::MoveVertex& move);
    libvgcode::EMoveType convertMoveType(Slic3r::EMoveType type);
    libvgcode::EGCodeExtrusionRole convertExtrusionRole(Slic3r::GCodeExtrusionRole role);
    
    // Create default color palettes
    std::vector<std::array<float, 4>> createDefaultToolColors();
    std::vector<std::array<float, 4>> createDefaultColorPrintColors();
}; 