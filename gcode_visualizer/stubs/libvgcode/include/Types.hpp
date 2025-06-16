#pragma once
#include <array>

namespace libvgcode {
    
// Basic matrix type for transformations (16 elements in column-major order)
using Mat4x4 = std::array<float, 16>;

// Enums for G-code visualization
enum class EMoveType {
    Noop = 0,
    Retract,
    Unretract,
    Seam,
    ToolChange,
    ColorChange,
    PausePrint,
    CustomGCode,
    Travel,
    Wipe,
    Extrude
};

enum class EGCodeExtrusionRole {
    None = 0,
    Perimeter,
    ExternalPerimeter,
    OverhangPerimeter,
    InternalInfill,
    SolidInfill,
    TopSolidInfill,
    Ironing,
    BridgeInfill,
    GapFill,
    Skirt,
    SupportMaterial,
    SupportMaterialInterface,
    WipeTower,
    Custom
};

} // namespace libvgcode 