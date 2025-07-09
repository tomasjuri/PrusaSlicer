# Printer Configuration System Usage Guide

This guide shows how to use the new printer configuration system in gcode_visualizer that automatically loads STL bed models and SVG textures based on printer model names.

## Quick Start

Instead of manually specifying STL and SVG file paths, you can now simply specify a printer model:

```cpp
#include "BedRenderer.hpp"

// Old way (manual paths)
BedRenderer renderer;
renderer.initialize("/path/to/mk4_bed.stl", "/path/to/mk4s.svg");

// New way (automatic configuration)
BedRenderer renderer;
renderer.initializeFromPrinterModel("MK4S");
```

## Supported Printer Models

The system reads configuration from `resources/profiles/PrusaResearch.ini` and automatically supports all Prusa printers:

### Popular Models:
- **MK4S** - `bed_model = mk4_bed.stl`, `bed_texture = mk4s.svg`
- **MK4IS** - `bed_model = mk4_bed.stl`, `bed_texture = mk4.svg`
- **MK3S** - `bed_model = mk3_bed.stl`, `bed_texture = mk3.svg`
- **XL** - `bed_model = xl_bed.stl`, `bed_texture = xl.svg`
- **MINI** - `bed_model = mini_bed.stl`, `bed_texture = mini.svg`
- **COREONE** - `bed_model = coreone_bed.stl`, `bed_texture = coreone.svg`

### Full List:
Run the example program to see all available models, or check the configuration file.

## Usage Examples

### Example 1: Basic Usage
```cpp
#include "BedRenderer.hpp"

int main() {
    BedRenderer renderer;
    
    // Initialize for MK4S printer
    if (renderer.initializeFromPrinterModel("MK4S")) {
        std::cout << "MK4S bed loaded successfully!" << std::endl;
        // The renderer now has:
        // - mk4_bed.stl loaded as 3D model
        // - mk4s.svg loaded as texture
        
        // Render in your loop
        renderer.render(view_matrix, projection_matrix);
    }
    
    return 0;
}
```

### Example 2: Custom Configuration File
```cpp
#include "BedRenderer.hpp"

int main() {
    BedRenderer renderer;
    
    // Use custom configuration file
    std::string config_path = "/path/to/my/printer_profiles.ini";
    
    if (renderer.initializeFromPrinterModel("MK4S", config_path)) {
        std::cout << "Loaded from custom config!" << std::endl;
    }
    
    return 0;
}
```

### Example 3: Fallback Handling
```cpp
#include "BedRenderer.hpp"

int main() {
    BedRenderer renderer;
    
    // Try to load specific model, fall back to default if not found
    if (!renderer.initializeFromPrinterModel("UNKNOWN_MODEL")) {
        std::cout << "Model not found, using default rendering" << std::endl;
        // The system automatically falls back to procedural bed rendering
    }
    
    return 0;
}
```

### Example 4: Using PrinterConfig Directly
```cpp
#include "PrinterConfig.hpp"

int main() {
    PrinterConfig config;
    
    if (config.loadConfig("resources/profiles/PrusaResearch.ini")) {
        // Get specific paths
        std::string stl_path = config.getBedModelPath("MK4S");
        std::string svg_path = config.getBedTexturePath("MK4S");
        
        std::cout << "STL: " << stl_path << std::endl;
        std::cout << "SVG: " << svg_path << std::endl;
        
        // Get model info
        PrinterModelInfo* info = config.findPrinterModel("MK4S");
        if (info) {
            std::cout << "Name: " << info->name << std::endl;
            std::cout << "Family: " << info->family << std::endl;
        }
        
        // List all available models
        auto models = config.getAvailableModels();
        for (const auto& model : models) {
            std::cout << "Available: " << model << std::endl;
        }
    }
    
    return 0;
}
```

## Configuration File Format

The system reads standard PrusaSlicer `.ini` configuration files:

```ini
[printer_model:MK4S]
name = Original Prusa MK4S
variants = HF0.4; HF0.5; HF0.6; HF0.8; 0.25; 0.3; 0.4; 0.5; 0.6; 0.8
technology = FFF
family = MK4
bed_model = mk4_bed.stl
bed_texture = mk4s.svg
thumbnail = MK4S_thumbnail.png
default_materials = Prusament PLA @MK4S HF0.4; Prusament PLA Blend @MK4S; ...
```

### Key Fields:
- `bed_model` - STL file for 3D bed geometry
- `bed_texture` - SVG file for bed texture/grid pattern
- `name` - Human-readable printer name
- `family` - Printer family (MK4, MK3, XL, etc.)
- `technology` - FFF or SLA

## File Path Resolution

The system automatically resolves file paths:

1. **Relative to config file**: `resources/profiles/PrusaResearch/mk4_bed.stl`
2. **Fallback**: `resources/profiles/mk4_bed.stl`
3. **Absolute paths**: Used as-is if provided

## Error Handling

The system provides robust error handling:

- **Missing config file**: Falls back to default rendering
- **Unknown printer model**: Lists available models and falls back
- **Missing STL/SVG files**: Warns and continues with available resources
- **Invalid file formats**: Graceful fallback to procedural rendering

## Integration with Existing Code

The new system is fully backward-compatible:

```cpp
// This still works
renderer.initialize("manual_bed.stl", "manual_texture.svg");

// But this is easier and more maintainable
renderer.initializeFromPrinterModel("MK4S");
```

## Benefits

1. **Automatic Configuration**: No need to manually specify file paths
2. **Consistency**: Uses the same profiles as PrusaSlicer
3. **Easy Updates**: Adding new printers just requires updating the .ini file
4. **Robust**: Graceful fallbacks when files are missing
5. **Maintainable**: Centralized configuration instead of hardcoded paths

## Testing

Run the example program to test the system:

```bash
./example_printer_config
```

This will:
1. Load the configuration file
2. List all available printer models
3. Test specific model lookups
4. Initialize BedRenderer with different models
5. Verify rendering functionality

## Output Example

When you use `initializeFromPrinterModel("MK4S")`, you'll see:

```
Initializing bed renderer for printer model: MK4S
Loading printer configuration from: ../resources/profiles/PrusaResearch.ini
Loaded 25 printer models from configuration
  - COREONE (Prusa CORE One)
  - MK4S (Original Prusa MK4S)
  - MK3S (Original Prusa i3 MK3S && MK3S+)
  - ...
Parsed printer model: MK4S (Original Prusa MK4S)
  Bed model: mk4_bed.stl
  Bed texture: mk4s.svg
Found configuration for MK4S:
  STL Model: /path/to/resources/profiles/PrusaResearch/mk4_bed.stl
  SVG Texture: /path/to/resources/profiles/PrusaResearch/mk4s.svg
Loading bed model: /path/to/resources/profiles/PrusaResearch/mk4_bed.stl
Loaded bed model with 1250 vertices
Professional bed renderer initialized successfully
```

This gives you the exact PrusaSlicer bed visualization for the specified printer model! 