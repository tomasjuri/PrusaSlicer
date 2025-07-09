#include <iostream>
#include <string>
#include "GCodeVisualizerApp.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cout << "Usage: " << argv[0] << " <gcode_file> [printer_model]" << std::endl;
        std::cout << "Example: " << argv[0] << " example.gcode" << std::endl;
        std::cout << "Example: " << argv[0] << " example.gcode MK4S" << std::endl;
        std::cout << "\nSupported printer models: MK4S, MK3S, XL, MINI, etc." << std::endl;
        std::cout << "If no printer model is specified, MK4S will be used as default." << std::endl;
        return 1;
    }

    std::string gcode_file = argv[1];
    std::string printer_model = (argc == 3) ? argv[2] : "MK4S";  // Default to MK4S
    
    std::cout << "G-Code Visualizer with Real SVG Parsing (NanoSVG)" << std::endl;
    std::cout << "Loading G-code file: " << gcode_file << std::endl;
    std::cout << "Using printer model: " << printer_model << std::endl;

    try {
        GCodeVisualizerApp app;
        
        if (!app.initialize(printer_model)) {
            std::cerr << "Failed to initialize application with " << printer_model << std::endl;
            return 1;
        }

        if (!app.loadGCode(gcode_file)) {
            std::cerr << "Failed to load G-code file: " << gcode_file << std::endl;
            return 1;
        }

        if (!app.renderAndSave()) {
            std::cerr << "Failed to render and save visualization" << std::endl;
            return 1;
        }

        std::cout << "G-code visualization completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 