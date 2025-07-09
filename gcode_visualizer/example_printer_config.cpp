#include <iostream>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "BedRenderer.hpp"
#include "PrinterConfig.hpp"

// Simple matrix implementation for the example
struct Mat4x4 {
    float m_data[16];
    
    Mat4x4() {
        // Initialize as identity matrix
        for (int i = 0; i < 16; i++) m_data[i] = 0.0f;
        m_data[0] = m_data[5] = m_data[10] = m_data[15] = 1.0f;
    }
    
    const float* data() const { return m_data; }
};

int main(int argc, char* argv[]) {
    std::cout << "=== Printer Configuration Example ===" << std::endl;
    
    // Initialize OpenGL context (minimal setup)
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hidden window for this example
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Printer Config Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }
    
    // Test the configuration system directly
    PrinterConfig config;
    std::string config_path = "../resources/profiles/PrusaResearch.ini";
    
    // Use command line argument if provided
    if (argc > 1) {
        config_path = argv[1];
        std::cout << "Using config file from command line: " << config_path << std::endl;
    } else {
        std::cout << "Using default config file: " << config_path << std::endl;
        std::cout << "Tip: You can specify a custom config file as: ./example_printer_config <path_to_config.ini>" << std::endl;
    }
    
    std::cout << "\n1. Testing configuration loading..." << std::endl;
    if (config.loadConfig(config_path)) {
        std::cout << "✓ Configuration loaded successfully!" << std::endl;
        
        // List available models
        auto models = config.getAvailableModels();
        std::cout << "\nAvailable printer models (" << models.size() << "):" << std::endl;
        for (const auto& model : models) {
            std::cout << "  - " << model << std::endl;
        }
    } else {
        std::cout << "✗ Failed to load configuration!" << std::endl;
        return -1;
    }
    
    // Test specific printer models
    std::vector<std::string> test_models = {"MK4S", "MK3S", "XL", "MINI"};
    
    std::cout << "\n2. Testing printer model lookup..." << std::endl;
    for (const auto& model : test_models) {
        PrinterModelInfo* info = config.findPrinterModel(model);
        if (info) {
            std::cout << "\n✓ Found " << model << " (" << info->name << "):" << std::endl;
            std::cout << "    Bed Model: " << info->bed_model << std::endl;
            std::cout << "    Bed Texture: " << info->bed_texture << std::endl;
            std::cout << "    Technology: " << info->technology << std::endl;
            std::cout << "    Family: " << info->family << std::endl;
            
            // Test path resolution
            std::string stl_path = config.getBedModelPath(model);
            std::string svg_path = config.getBedTexturePath(model);
            std::cout << "    Resolved STL: " << stl_path << std::endl;
            std::cout << "    Resolved SVG: " << svg_path << std::endl;
        } else {
            std::cout << "✗ " << model << " not found" << std::endl;
        }
    }
    
    // Test BedRenderer integration
    std::cout << "\n3. Testing BedRenderer integration..." << std::endl;
    BedRenderer renderer;
    
    // Test with MK4S
    std::cout << "\nTesting MK4S bed rendering:" << std::endl;
    if (renderer.initializeFromPrinterModel("MK4S")) {
        std::cout << "✓ MK4S bed renderer initialized successfully!" << std::endl;
        
        // Create simple matrices for testing
        Mat4x4 view_matrix;
        Mat4x4 projection_matrix;
        
        // Test rendering (this would normally be in a render loop)
        std::cout << "Testing render call..." << std::endl;
        renderer.render(view_matrix, projection_matrix);
        std::cout << "✓ Render call completed without errors!" << std::endl;
        
    } else {
        std::cout << "✗ Failed to initialize MK4S bed renderer!" << std::endl;
    }
    
    // Test with unknown model
    std::cout << "\nTesting unknown model 'UNKNOWN':" << std::endl;
    BedRenderer renderer2;
    if (renderer2.initializeFromPrinterModel("UNKNOWN")) {
        std::cout << "✓ Unknown model fell back to default rendering" << std::endl;
    } else {
        std::cout << "✗ Failed to initialize with unknown model" << std::endl;
    }
    
    std::cout << "\n=== Example completed successfully! ===" << std::endl;
    
    // Cleanup
    renderer.cleanup();
    renderer2.cleanup();
    glfwTerminate();
    
    return 0;
} 