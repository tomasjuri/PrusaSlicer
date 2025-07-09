#pragma once

#include <string>
#include <map>
#include <vector>

// Structure to hold printer model information
struct PrinterModelInfo {
    std::string name;
    std::string bed_model;      // STL file path
    std::string bed_texture;    // SVG file path
    std::string thumbnail;
    std::vector<std::string> variants;
    std::string technology;
    std::string family;
    std::vector<std::pair<float, float>> bed_shape;  // Bed shape points
    
    PrinterModelInfo() = default;
};

// Configuration parser for PrusaSlicer-style .ini files
class PrinterConfig {
public:
    PrinterConfig();
    ~PrinterConfig();
    
    // Load configuration from .ini file (e.g., "resources/profiles/PrusaResearch.ini")
    bool loadConfig(const std::string& config_file_path);
    
    // Find printer model by key (e.g., "MK4S", "MK3S", "XL", etc.)
    PrinterModelInfo* findPrinterModel(const std::string& model_key);
    
    // Get resolved file paths for bed model and texture
    std::string getBedModelPath(const std::string& model_key);
    std::string getBedTexturePath(const std::string& model_key);
    
    // Get bed shape points for a printer model
    std::vector<std::pair<float, float>> getBedShape(const std::string& model_key);
    
    // Set the resources directory (where profiles are located)
    void setResourcesDir(const std::string& resources_dir);
    
    // Get list of available printer models
    std::vector<std::string> getAvailableModels() const;
    
private:
    std::map<std::string, PrinterModelInfo> m_printer_models;
    std::string m_resources_dir;
    std::string m_vendor_id;  // e.g., "PrusaResearch"
    
    // Parse a single line from .ini file
    bool parseLine(const std::string& line, std::string& section, std::string& key, std::string& value);
    
    // Parse printer model section
    void parsePrinterModelSection(const std::string& model_id, const std::map<std::string, std::string>& properties);
    
    // Resolve full path for resource files
    std::string resolveResourcePath(const std::string& filename) const;
    
    // Trim whitespace from string
    std::string trim(const std::string& str) const;
    
    // Parse bed shape from string (format: "0x0,250x0,250x210,0x210")
    std::vector<std::pair<float, float>> parseBedShape(const std::string& bed_shape_str) const;
}; 