![PrusaSlicer logo](/resources/icons/PrusaSlicer.png?raw=true)

# PrusaSlicer

You may want to check the [PrusaSlicer project page](https://www.prusa3d.com/prusaslicer/).
Prebuilt Windows, OSX and Linux binaries are available through the [git releases page](https://github.com/prusa3d/PrusaSlicer/releases) or from the [Prusa3D downloads page](https://www.prusa3d.com/drivers/). There are also [3rd party Linux builds available](https://github.com/prusa3d/PrusaSlicer/wiki/PrusaSlicer-on-Linux---binary-distributions).

PrusaSlicer takes 3D models (STL, OBJ, AMF) and converts them into G-code
instructions for FFF printers or PNG layers for mSLA 3D printers. It's
compatible with any modern printer based on the RepRap toolchain, including all
those based on the Marlin, Prusa, Sprinter and Repetier firmware. It also works
with Mach3, LinuxCNC and Machinekit controllers.

PrusaSlicer is based on [Slic3r](https://github.com/Slic3r/Slic3r) by Alessandro Ranellucci and the RepRap community.

See the [project homepage](https://www.prusa3d.com/slic3r-prusa-edition/) and
the [documentation directory](doc/) for more information.

### What language is it written in?

All user facing code is written in C++.
The slicing core is the `libslic3r` library, which can be built and used in a standalone way.
The command line interface is a thin wrapper over `libslic3r`.

### What are PrusaSlicer's main features?

Key features are:

* **multi-platform** (Linux/Mac/Win) and packaged as standalone-app with no dependencies required
* complete **command-line interface** to use it with no GUI
* multi-material **(multiple extruders)** object printing
* multiple G-code flavors supported (RepRap, Makerbot, Mach3, Machinekit etc.)
* ability to plate **multiple objects having distinct print settings**
* **multithread** processing
* **STL auto-repair** (tolerance for broken models)
* wide automated unit testing

Other major features are:

* combine infill every 'n' perimeters layer to speed up printing
* **3D preview** (including multi-material files)
* **multiple layer heights** in a single print
* **spiral vase** mode for bumpless vases
* fine-grained configuration of speed, acceleration, extrusion width
* several infill patterns including honeycomb, spirals, Hilbert curves
* support material, raft, brim, skirt
* **standby temperature** and automatic wiping for multi-extruder printing
* [customizable **G-code macros**](https://github.com/prusa3d/PrusaSlicer/wiki/PrusaSlicer-Macro-Language) and output filename with variable placeholders
* support for **post-processing scripts**
* **cooling logic** controlling fan speed and dynamic print speed

### Development

If you want to compile the source yourself, follow the instructions on one of
these documentation pages:
* [Linux](doc/How%20to%20build%20-%20Linux%20et%20al.md)
* [macOS](doc/How%20to%20build%20-%20Mac%20OS.md)
* [Windows](doc/How%20to%20build%20-%20Windows.md)

### Can I help?

Sure! You can do the following to find things that are available to help with:
* Add an [issue](https://github.com/prusa3d/PrusaSlicer/issues) to the github tracker if it isn't already present.
* Look at [issues labeled "volunteer needed"](https://github.com/prusa3d/PrusaSlicer/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3A%22volunteer+needed%22)

### What's PrusaSlicer license?

PrusaSlicer is licensed under the _GNU Affero General Public License, version 3_.
The PrusaSlicer is originally based on Slic3r by Alessandro Ranellucci.

### How can I use PrusaSlicer from the command line?

Please refer to the [Command Line Interface](https://github.com/prusa3d/PrusaSlicer/wiki/Command-Line-Interface) wiki page.

# G-Code Visualizer

A standalone G-code visualization application built using PrusaSlicer's existing G-code processing and visualization components.

## Overview

This application demonstrates how to reuse PrusaSlicer's powerful G-code processing and visualization components to create a custom visualization tool. It:

- Loads G-code files using PrusaSlicer's `GCodeProcessor`
- Visualizes the toolpaths using the `libvgcode` library
- Renders the print bed surface similar to PrusaSlicer's 3D view
- Provides a top-down camera view positioned 10cm from the build plate
- Exports the visualization as a JPEG image

## Architecture

The application consists of several key components:

### Core Components

1. **GCodeVisualizerApp** - Main application class that orchestrates all components
2. **GCodeConverter** - Converts PrusaSlicer's GCodeProcessor results to libvgcode format  
3. **CameraController** - Manages camera positioning and view/projection matrices
4. **BedRenderer** - Renders the print bed grid and surface using OpenGL
5. **ImageExporter** - Exports the rendered frame to JPEG format

### PrusaSlicer Integration

The application leverages these existing PrusaSlicer components:

- **libslic3r/GCode/GCodeProcessor** - Parses and processes G-code files
- **libvgcode** - High-performance G-code visualization library  
- **3DBed rendering logic** - Adapted from `src/slic3r/GUI/3DBed.cpp`

## Key Features

### G-Code Processing
- Supports both ASCII (.gcode) and binary (.bgcode) G-code formats
- Parses all move types (extrusion, travel, retraction, etc.)
- Extracts extrusion roles (perimeter, infill, support, etc.)  
- Preserves timing and temperature information

### Visualization
- 3D toolpath rendering with proper colors for different extrusion roles
- Print bed grid visualization (10mm spacing by default)
- Top-down camera view positioned 10cm above the build plate
- Offscreen rendering for headless operation

### Print Bed Rendering
Based on PrusaSlicer's 3DBed implementation:
- Grid lines every 10mm (configurable)
- Standard Prusa bed size (250x210mm) by default
- Proper OpenGL depth testing for correct layering

## Usage

```bash
# Basic usage
./gcode_visualizer path/to/your/file.gcode

# The application will create gcode_visualization.jpg in the current directory
```

### Example with provided test file
```bash
./gcode_visualizer 3dbenchy_0.4n_0.2mm_PETG_MK4IS_45m.gcode
```

## Camera Positioning

The camera is positioned for an optimal top-down view:
- **Position**: 10cm (100mm) directly above the bed center
- **Target**: Center of the build plate (125, 105, 0)  
- **Up vector**: Y-axis pointing "up" in the view
- **Field of view**: 45 degrees with proper aspect ratio

This provides a clear view of the entire print while maintaining enough distance to see all details.

## Output

The application generates:
- **gcode_visualization.jpg** - High-quality JPEG image (1920x1080) of the visualization
- Console output showing:
  - G-code loading progress
  - Number of moves parsed  
  - OpenGL initialization status
  - Rendering completion confirmation

## Building

This application requires the full PrusaSlicer build environment due to its dependencies on:

- libslic3r (G-code processing)
- libvgcode (visualization)  
- OpenGL libraries
- GLFW for context creation
- Various image processing libraries

### Prerequisites

1. Install PrusaSlicer build dependencies
2. Build PrusaSlicer first to ensure all libraries are available
3. Then build the G-code visualizer

### Build Commands

```bash
# From the PrusaSlicer root directory
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) gcode_visualizer
```

## Code Structure

```
gcode_visualizer/
├── main.cpp                 # Entry point
├── GCodeVisualizerApp.cpp   # Main application logic  
├── GCodeVisualizerApp.hpp   # Main application header
├── GCodeConverter.cpp       # Data format conversion
├── GCodeConverter.hpp       # Converter interface
├── CameraController.cpp     # Camera management
├── CameraController.hpp     # Camera interface
├── BedRenderer.cpp          # Print bed rendering
├── BedRenderer.hpp          # Bed renderer interface
├── ImageExporter.cpp        # JPEG export functionality
└── ImageExporter.hpp        # Export interface
```

## Extension Points

This application can be easily extended:

### Different Camera Angles
Modify `CameraController::setupTopView()` to support:
- Side views
- Isometric views  
- Multiple angle captures

### Custom Bed Sizes
Adjust `BedRenderer::initialize()` parameters for different printer models:
- Bed dimensions
- Grid spacing
- Grid colors

### Additional Export Formats
Extend `ImageExporter` to support:
- PNG format
- Different resolutions
- Multiple output files

### Animation Support
The rendering pipeline can be extended to create:
- Layer-by-layer animations
- Print progression videos
- Interactive views

## Technical Implementation Details

### Offscreen Rendering
The application uses OpenGL framebuffer objects for headless operation:
- Creates invisible GLFW window
- Renders to texture using FBO
- Reads pixels for image export

### Matrix Mathematics  
Camera matrices are computed manually for precise control:
- Look-at matrix for camera positioning
- Perspective projection with proper aspect ratio
- Coordinate system transformations

### Memory Management
Proper resource cleanup ensures no memory leaks:
- OpenGL objects are explicitly deleted
- Smart pointers manage component lifetimes
- RAII principles throughout

This demonstrates how PrusaSlicer's modular architecture enables powerful customization and extension for specialized use cases.
