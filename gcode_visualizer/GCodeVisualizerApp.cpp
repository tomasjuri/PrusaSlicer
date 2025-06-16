#include "GCodeVisualizerApp.hpp"
#include <iostream>
#include <filesystem>

GCodeVisualizerApp::GCodeVisualizerApp() 
    : m_window(nullptr)
    , m_gl_initialized(false)
    , m_gcode_loaded(false)
{
}

GCodeVisualizerApp::~GCodeVisualizerApp() {
    cleanup();
}

bool GCodeVisualizerApp::initialize() {
    std::cout << "Initializing G-Code Visualizer..." << std::endl;
    
    // Initialize GLFW
    if (!initializeGLFW()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return false;
    }
    
    // Initialize OpenGL
    if (!initializeOpenGL()) {
        cleanup();
        return false;
    }
    
    // Create offscreen framebuffer
    if (!createOffscreenFramebuffer()) {
        cleanup();
        return false;
    }
    
    // Create our simple G-code parser
    m_gcode_parser = std::make_unique<SimpleGCode::GCodeParser>();
    
    // Create other components
    m_camera_controller = std::make_unique<CameraController>();
    m_image_exporter = std::make_unique<ImageExporter>(WINDOW_WIDTH, WINDOW_HEIGHT);
    m_bed_renderer = std::make_unique<BedRenderer>();
    m_test_cube_renderer = std::make_unique<TestCubeRenderer>();
    m_gcode_path_renderer = std::make_unique<GCodePathRenderer>();
    
    // Initialize components with Prusa MK4 bed assets
    std::string stl_path = "/Users/tomasjurica/projects/PrusaSlicer-1/resources/profiles/PrusaResearch/mk4_bed.stl";
    std::string svg_path = "/Users/tomasjurica/projects/PrusaSlicer-1/resources/profiles/PrusaResearch/mk4.svg";
    if (!m_bed_renderer->initialize(stl_path, svg_path)) {
        std::cerr << "Failed to initialize bed renderer with Prusa assets, using fallback" << std::endl;
        // Fallback to simple bed
        m_bed_renderer->initialize("", "");
    }
    m_test_cube_renderer->initialize();
    if (!m_gcode_path_renderer->initialize()) {
        std::cerr << "Failed to initialize G-code path renderer" << std::endl;
        return false;
    }
    
    std::cout << "Initialization complete!" << std::endl;
    return true;
}

bool GCodeVisualizerApp::loadGCode(const std::string& filename) {
    std::cout << "Loading G-code file: " << filename << std::endl;
    
    if (!std::filesystem::exists(filename)) {
        std::cerr << "G-code file does not exist: " << filename << std::endl;
        return false;
    }
    
    // Use our simple G-code parser
    if (!m_gcode_parser->parseFile(filename)) {
        std::cerr << "Failed to parse G-code file" << std::endl;
        return false;
    }
    
    const auto& moves = m_gcode_parser->getMoves();
    if (moves.empty()) {
        std::cerr << "No moves found in G-code file" << std::endl;
        return false;
    }
    
    // Print detailed analysis
    m_gcode_parser->printAnalysis();
    
    // Load moves into path renderer
    m_gcode_path_renderer->setGCodeMoves(moves);
    
    // Set up optimal camera view based on print bounds
    auto bounds = m_gcode_parser->getPrintBounds();
    if (bounds.has_value()) {
        auto [min_x, max_x, min_y, max_y, max_z] = bounds.value();
        std::cout << "Setting optimal camera view for print bounds..." << std::endl;
        m_camera_controller->setOptimalView(min_x, max_x, min_y, max_y, max_z, WINDOW_WIDTH, WINDOW_HEIGHT);
    } else {
        // Fallback to standard positioning 50cm from surface
        std::cout << "Using standard camera positioning..." << std::endl;
        m_camera_controller->setDistanceFromSurface(500.0f);  // 50cm
    }
    
    m_gcode_loaded = true;
    std::cout << "G-code loaded successfully!" << std::endl;
    return true;
}

bool GCodeVisualizerApp::renderAndSave() {
    std::cout << "Rendering G-code visualization..." << std::endl;
    
    // Set up camera for top view, 10cm from build plate
    setupTopViewCamera();
    
    // Bind offscreen framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    
    // Clear the framebuffer
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);  // Dark gray background
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Render the scene
    renderScene();
    
    // Ensure rendering is complete
    glFlush();
    glFinish();
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "OpenGL error after rendering: " << error << std::endl;
    }
    
    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Export the image
    std::string output_path = "gcode_visualization.png";
    std::cout << "Saving image to: " << output_path << std::endl;
    
    if (!m_image_exporter->saveTextureToJPEG(output_path, m_color_texture)) {
        std::cerr << "Failed to save image: " << output_path << std::endl;
        return false;
    }
    
    std::cout << "Saved: " << output_path << std::endl;
    return true;
}

void GCodeVisualizerApp::cleanup() {
    cleanupOffscreenFramebuffer();
    
    if (m_test_cube_renderer) {
        m_test_cube_renderer->cleanup();
        m_test_cube_renderer.reset();
    }
    
    if (m_gcode_path_renderer) {
        m_gcode_path_renderer->cleanup();
        m_gcode_path_renderer.reset();
    }
    
    m_bed_renderer.reset();
    m_image_exporter.reset();
    m_camera_controller.reset();
    m_gcode_parser.reset();
    
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    
    glfwTerminate();
    m_gl_initialized = false;
}



bool GCodeVisualizerApp::initializeGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return false;
    }
    
    // Set OpenGL version (3.3 Core Profile)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create invisible window for offscreen rendering
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    
    m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "G-Code Visualizer", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(m_window);
    return true;
}

bool GCodeVisualizerApp::initializeOpenGL() {
    // Initialize GLAD
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return false;
    }
    
    std::cout << "OpenGL " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    m_gl_initialized = true;
    return true;
}

bool GCodeVisualizerApp::createOffscreenFramebuffer() {
    // Generate framebuffer
    glGenFramebuffers(1, &m_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
    
    // Generate color texture
    glGenTextures(1, &m_color_texture);
    glBindTexture(GL_TEXTURE_2D, m_color_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color_texture, 0);
    
    // Generate depth renderbuffer
    glGenRenderbuffers(1, &m_depth_renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depth_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_renderbuffer);
    
    // Check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
        cleanupOffscreenFramebuffer();
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void GCodeVisualizerApp::cleanupOffscreenFramebuffer() {
    if (m_depth_renderbuffer) {
        glDeleteRenderbuffers(1, &m_depth_renderbuffer);
        m_depth_renderbuffer = 0;
    }
    if (m_color_texture) {
        glDeleteTextures(1, &m_color_texture);
        m_color_texture = 0;
    }
    if (m_framebuffer) {
        glDeleteFramebuffers(1, &m_framebuffer);
        m_framebuffer = 0;
    }
}

void GCodeVisualizerApp::setupTopViewCamera() {
    // Camera positioning is now done automatically after G-code loading
    // This method maintains the original camera setup if needed
    m_camera_controller->setTopView(WINDOW_WIDTH, WINDOW_HEIGHT);
}

void GCodeVisualizerApp::renderScene() {
    // Get camera matrices
    auto view_matrix = m_camera_controller->getViewMatrix();
    auto projection_matrix = m_camera_controller->getProjectionMatrix();
    
    // Render print bed
    m_bed_renderer->render(view_matrix, projection_matrix);
    
    // Render G-code paths (this is the main visualization!)
    if (m_gcode_loaded) {
        m_gcode_path_renderer->render(view_matrix, projection_matrix);
    }
    
    // Optionally render test cube (can be removed later)
    // m_test_cube_renderer->render(view_matrix, projection_matrix);
} 