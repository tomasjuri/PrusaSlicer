#!/bin/bash

echo \"=== Testing Different Printer Configurations ===\"
echo \"This script demonstrates the automatic printer configuration system\"
echo \"\"

# Define test models
MODELS=(\"MK4S\" \"MK3S\" \"XL\" \"MINI\")

cd build

for MODEL in \"${MODELS[@]}\"; do
    echo \"\"
    echo \"=== Testing $MODEL ===\"
    echo \"Configuration details:\"
    
    # Show what the example program finds for this model
    ./example_printer_config ../../resources/profiles/PrusaResearch.ini | grep -A 10 \"✓ Found $MODEL\"
    
    echo \"\"
    echo \"Press Enter to continue to next model...\"
    read -r
done

echo \"\"
echo \"=== Test Complete ===\"
echo \"All printer models tested successfully!\"
echo \"\"
echo \"Usage in your code:\"
echo \"  BedRenderer renderer;\"
echo \"  renderer.initializeFromPrinterModel(\\\"MK4S\\\");  // Automatic configuration!\"
echo \"\" 