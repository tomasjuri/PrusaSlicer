#include "ImageExporter.hpp"
#include <iostream>
#include <vector>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// For JPEG export, we'll use a simple method
// In a full implementation, you'd link against libjpeg or stb_image_write

ImageExporter::ImageExporter(int width, int height) 
    : m_width(width), m_height(height) {
}

bool ImageExporter::saveFramebufferToJPEG(const std::string& filename) {
    // Read pixels from framebuffer
    std::vector<unsigned char> pixels(m_width * m_height * 3);
    
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error reading pixels: " << error << std::endl;
        return false;
    }
    
    // Flip the image vertically (OpenGL has origin at bottom-left, images at top-left)
    flipImageVertically(pixels.data(), m_width, m_height, 3);
    
    // Save to PNG instead of JPEG
    return savePixelsToPNG(filename, pixels.data(), m_width, m_height);
}

bool ImageExporter::saveTextureToJPEG(const std::string& filename, GLuint texture_id) {
    // Bind texture and read its pixels
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    std::vector<unsigned char> pixels(m_width * m_height * 3);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error reading texture: " << error << std::endl;
        return false;
    }
    
    // Flip the image vertically
    flipImageVertically(pixels.data(), m_width, m_height, 3);
    
    // Save to PNG instead of JPEG
    return savePixelsToPNG(filename, pixels.data(), m_width, m_height);
}

void ImageExporter::setDimensions(int width, int height) {
    m_width = width;
    m_height = height;
}

bool ImageExporter::savePixelsToPNG(const std::string& filename, unsigned char* pixels, int width, int height) {
    // Change file extension to .png
    std::string png_filename = filename;
    size_t dot_pos = png_filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        png_filename = png_filename.substr(0, dot_pos) + ".png";
    } else {
        png_filename += ".png";
    }
    
    // Use stb_image_write to save PNG
    int result = stbi_write_png(png_filename.c_str(), width, height, 3, pixels, width * 3);
    
    if (result) {
        std::cout << "Image saved as PNG format: " << png_filename << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save PNG image: " << png_filename << std::endl;
        return false;
    }
}

void ImageExporter::flipImageVertically(unsigned char* pixels, int width, int height, int channels) {
    int row_size = width * channels;
    std::vector<unsigned char> temp_row(row_size);
    
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* top_row = pixels + y * row_size;
        unsigned char* bottom_row = pixels + (height - 1 - y) * row_size;
        
        // Swap rows
        std::memcpy(temp_row.data(), top_row, row_size);
        std::memcpy(top_row, bottom_row, row_size);
        std::memcpy(bottom_row, temp_row.data(), row_size);
    }
} 