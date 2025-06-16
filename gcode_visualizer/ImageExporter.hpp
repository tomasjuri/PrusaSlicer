#pragma once

#include <string>
#include "stubs/libvgcode/glad/include/glad/gl.h"

class ImageExporter {
public:
    ImageExporter(int width, int height);
    ~ImageExporter() = default;
    
    // Save the current framebuffer to a PNG file
    bool saveFramebufferToJPEG(const std::string& filename);
    
    // Save a specific texture to a PNG file
    bool saveTextureToJPEG(const std::string& filename, GLuint texture_id);
    
    // Set output dimensions
    void setDimensions(int width, int height);
    
private:
    int m_width;
    int m_height;
    
    // Helper methods
    bool savePixelsToPNG(const std::string& filename, unsigned char* pixels, int width, int height);
    void flipImageVertically(unsigned char* pixels, int width, int height, int channels);
}; 