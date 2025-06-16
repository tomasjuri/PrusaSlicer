#include <iostream>
#include <string>
#include "GCodeVisualizerApp.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <gcode_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " example.gcode" << std::endl;
        return 1;
    }

    std::string gcode_file = argv[1];
    
    std::cout << "G-Code Visualizer" << std::endl;
    std::cout << "Loading G-code file: " << gcode_file << std::endl;

    try {
        GCodeVisualizerApp app;
        
        if (!app.initialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
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