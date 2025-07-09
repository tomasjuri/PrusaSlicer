#include "PrinterConfig.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

PrinterConfig::PrinterConfig() {
    // Default to PrusaResearch if no vendor is specified
    m_vendor_id = "PrusaResearch";
}

PrinterConfig::~PrinterConfig() {
}

bool PrinterConfig::loadConfig(const std::string& config_file_path) {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_file_path << std::endl;
        return false;
    }
    
    std::cout << "Loading printer configuration from: " << config_file_path << std::endl;
    
    // Extract vendor ID from file path (e.g., "PrusaResearch.ini" -> "PrusaResearch")
    std::filesystem::path config_path(config_file_path);
    std::string filename = config_path.stem().string();
    if (!filename.empty()) {
        m_vendor_id = filename;
    }
    
    // Extract resources directory from config file path
    std::filesystem::path parent_dir = config_path.parent_path();
    if (m_resources_dir.empty()) {
        m_resources_dir = parent_dir.string();
    }
    
    std::string line;
    std::string current_section;
    std::map<std::string, std::string> current_properties;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Check for section header [printer_model:MODELNAME]
        if (line[0] == '[' && line.back() == ']') {
            // Process previous section if it was a printer model
            if (!current_section.empty() && current_section.find("printer_model:") == 0) {
                std::string model_id = current_section.substr(14); // Remove "printer_model:"
                parsePrinterModelSection(model_id, current_properties);
            }
            
            // Start new section
            current_section = line.substr(1, line.length() - 2); // Remove [ and ]
            current_properties.clear();
            continue;
        }
        
        // Parse key-value pairs
        std::string section_dummy, key, value;
        if (parseLine(line, section_dummy, key, value)) {
            current_properties[key] = value;
        }
    }
    
    // Process the last section if it was a printer model
    if (!current_section.empty() && current_section.find("printer_model:") == 0) {
        std::string model_id = current_section.substr(14);
        parsePrinterModelSection(model_id, current_properties);
    }
    
    file.close();
    
    std::cout << "Loaded " << m_printer_models.size() << " printer models from configuration" << std::endl;
    
    // Print available models for debugging
    for (const auto& pair : m_printer_models) {
        std::cout << "  - " << pair.first << " (" << pair.second.name << ")" << std::endl;
    }
    
    return !m_printer_models.empty();
}

PrinterModelInfo* PrinterConfig::findPrinterModel(const std::string& model_key) {
    auto it = m_printer_models.find(model_key);
    if (it != m_printer_models.end()) {
        return &it->second;
    }
    return nullptr;
}

std::string PrinterConfig::getBedModelPath(const std::string& model_key) {
    PrinterModelInfo* model = findPrinterModel(model_key);
    if (model && !model->bed_model.empty()) {
        return resolveResourcePath(model->bed_model);
    }
    return "";
}

std::string PrinterConfig::getBedTexturePath(const std::string& model_key) {
    PrinterModelInfo* model = findPrinterModel(model_key);
    if (model && !model->bed_texture.empty()) {
        return resolveResourcePath(model->bed_texture);
    }
    return "";
}

std::vector<std::pair<float, float>> PrinterConfig::getBedShape(const std::string& model_key) {
    PrinterModelInfo* model = findPrinterModel(model_key);
    if (model) {
        return model->bed_shape;
    }
    return {};
}

void PrinterConfig::setResourcesDir(const std::string& resources_dir) {
    m_resources_dir = resources_dir;
}

std::vector<std::string> PrinterConfig::getAvailableModels() const {
    std::vector<std::string> models;
    for (const auto& pair : m_printer_models) {
        models.push_back(pair.first);
    }
    return models;
}

bool PrinterConfig::parseLine(const std::string& line, std::string& section, std::string& key, std::string& value) {
    size_t equals_pos = line.find('=');
    if (equals_pos == std::string::npos) {
        return false;
    }
    
    key = trim(line.substr(0, equals_pos));
    value = trim(line.substr(equals_pos + 1));
    
    return !key.empty();
}

void PrinterConfig::parsePrinterModelSection(const std::string& model_id, const std::map<std::string, std::string>& properties) {
    PrinterModelInfo info;
    
    auto it = properties.find("name");
    if (it != properties.end()) {
        info.name = it->second;
    }
    
    it = properties.find("bed_model");
    if (it != properties.end()) {
        info.bed_model = it->second;
    }
    
    it = properties.find("bed_texture");
    if (it != properties.end()) {
        info.bed_texture = it->second;
    }
    
    it = properties.find("thumbnail");
    if (it != properties.end()) {
        info.thumbnail = it->second;
    }
    
    it = properties.find("technology");
    if (it != properties.end()) {
        info.technology = it->second;
    }
    
    it = properties.find("family");
    if (it != properties.end()) {
        info.family = it->second;
    }
    
    it = properties.find("bed_shape");
    if (it != properties.end()) {
        info.bed_shape = parseBedShape(it->second);
    }
    
    it = properties.find("variants");
    if (it != properties.end()) {
        // Parse variants (semicolon-separated)
        std::string variants_str = it->second;
        std::stringstream ss(variants_str);
        std::string variant;
        while (std::getline(ss, variant, ';')) {
            variant = trim(variant);
            if (!variant.empty()) {
                info.variants.push_back(variant);
            }
        }
    }
    
    // Store the printer model
    m_printer_models[model_id] = info;
    
    std::cout << "Parsed printer model: " << model_id << " (" << info.name << ")" << std::endl;
    if (!info.bed_model.empty()) {
        std::cout << "  Bed model: " << info.bed_model << std::endl;
    }
    if (!info.bed_texture.empty()) {
        std::cout << "  Bed texture: " << info.bed_texture << std::endl;
    }
}

std::string PrinterConfig::resolveResourcePath(const std::string& filename) const {
    if (filename.empty()) {
        return "";
    }
    
    // If it's already an absolute path, return as-is
    std::filesystem::path file_path(filename);
    if (file_path.is_absolute()) {
        return filename;
    }
    
    // Construct path: resources_dir/vendor_id/filename
    std::filesystem::path full_path = std::filesystem::path(m_resources_dir) / m_vendor_id / filename;
    
    // Check if file exists
    if (std::filesystem::exists(full_path)) {
        return full_path.string();
    }
    
    // If not found, try without vendor subdirectory
    full_path = std::filesystem::path(m_resources_dir) / filename;
    if (std::filesystem::exists(full_path)) {
        return full_path.string();
    }
    
    std::cerr << "Warning: Resource file not found: " << filename << std::endl;
    std::cerr << "  Tried: " << std::filesystem::path(m_resources_dir) / m_vendor_id / filename << std::endl;
    std::cerr << "  Tried: " << std::filesystem::path(m_resources_dir) / filename << std::endl;
    
    return filename; // Return original filename as fallback
}

std::string PrinterConfig::trim(const std::string& str) const {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::vector<std::pair<float, float>> PrinterConfig::parseBedShape(const std::string& bed_shape_str) const {
    std::vector<std::pair<float, float>> points;
    
    if (bed_shape_str.empty()) {
        return points;
    }
    
    // Parse format like: "0x0,250x0,250x210,0x210"
    std::stringstream ss(bed_shape_str);
    std::string point_str;
    
    while (std::getline(ss, point_str, ',')) {
        point_str = trim(point_str);
        if (point_str.empty()) {
            continue;
        }
        
        // Find 'x' separator
        size_t x_pos = point_str.find('x');
        if (x_pos == std::string::npos) {
            continue;
        }
        
        try {
            float x = std::stof(point_str.substr(0, x_pos));
            float y = std::stof(point_str.substr(x_pos + 1));
            points.push_back({x, y});
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse bed shape point: " << point_str << std::endl;
        }
    }
    
    std::cout << "Parsed bed shape with " << points.size() << " points" << std::endl;
    for (const auto& point : points) {
        std::cout << "  Point: (" << point.first << ", " << point.second << ")" << std::endl;
    }
    
    return points;
} 