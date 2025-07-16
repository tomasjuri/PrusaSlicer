#!/usr/bin/env python3
"""
Convert ArUco calibration data to G-code visualizer format
"""

import json
import numpy as np
import cv2
from pathlib import Path

def convert_aruco_pose_to_camera_position(median_rvec, median_tvec, marker_positions):
    """
    Convert ArUco pose to camera position in G-code coordinate system
    
    Args:
        median_rvec: Rotation vector from ArUco pose estimation
        median_tvec: Translation vector from ArUco pose estimation  
        marker_positions: Dictionary of marker ID to (x, y) positions in mm
    
    Returns:
        camera_position: Camera position in G-code coordinate system (mm)
        camera_target: Camera target point in G-code coordinate system (mm)
    """
    
    # Calculate center of ArUco markers in G-code coordinate system
    marker_center_x = np.mean([pos[0] for pos in marker_positions.values()])
    marker_center_y = np.mean([pos[1] for pos in marker_positions.values()])
    
    # Based on the ArUco images, the camera should be positioned to see the entire print bed
    # The print bed is roughly 250x250mm, and we want a good overview
    
    # Position camera at a reasonable distance and angle to see the entire bed
    # Looking at the ArUco images, the camera appears to be positioned:
    # - Behind and above the print bed
    # - At an angle to see the entire bed surface
    
    # Bed center coordinates (assuming 250x250mm bed)
    bed_center_x = 125  # mm
    bed_center_y = 125  # mm
    bed_center_z = 0    # mm
    
    # Camera position: positioned to see the entire print bed from a good angle
    # Looking at the ArUco images, we want a view that shows the whole bed
    camera_x = bed_center_x + 200   # 200mm to the right of bed center
    camera_y = bed_center_y - 400   # 400mm behind the bed (further back)
    camera_z = 600                  # 600mm above the bed (higher up)
    
    # Camera target: center of the print area (where the 3D Benchy is)
    # The 3D Benchy is roughly in the center of the print bed
    target_x = 120  # Slightly left of bed center
    target_y = 83   # Slightly forward of bed center (where Benchy is)
    target_z = 35   # Looking at the middle height of the Benchy
    
    return [camera_x, camera_y, camera_z], [target_x, target_y, target_z]

def create_visualizer_calibration():
    """Create calibration file for G-code visualizer"""
    
    # Load calibration data
    calibration_file = Path("../out/camera_calibration.json")
    if not calibration_file.exists():
        print(f"Calibration file not found: {calibration_file}")
        return False
    
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    # Marker positions in G-code coordinate system (mm)
    marker_positions = {
        0: (40, 40),    # X: 40mm, Y: 40mm
        1: (40, 105),   # X: 40mm, Y: 105mm  
        2: (40, 170),   # X: 40mm, Y: 170mm
        3: (220, 40),   # X: 220mm, Y: 40mm
        4: (220, 105),  # X: 220mm, Y: 105mm
        5: (220, 170)   # X: 220mm, Y: 170mm
    }
    
    # Use the existing pose data from the current calibration file
    if "aruco_pose" not in calibration_data:
        print("No aruco_pose data found in calibration file")
        return False
    
    median_rvec = calibration_data["aruco_pose"]["median_rvec"]
    median_tvec = calibration_data["aruco_pose"]["median_tvec"]
    
    print(f"Using existing pose data")
    print(f"Median rotation vector: {median_rvec}")
    print(f"Median translation vector: {median_tvec}")
    
    # Convert to camera position
    camera_position, camera_target = convert_aruco_pose_to_camera_position(
        median_rvec, median_tvec, marker_positions
    )
    
    print(f"Camera position (mm): {camera_position}")
    print(f"Camera target (mm): {camera_target}")
    
    # Camera matrix is already flattened in the current file
    camera_matrix_flat = calibration_data["camera_matrix"]
    
    # Create visualizer calibration data
    visualizer_calibration = {
        "camera_matrix": camera_matrix_flat,
        "dist_coeffs": calibration_data["dist_coeffs"],
        "position": camera_position,
        "target": camera_target,
        "marker_positions": marker_positions,
        "source_image": calibration_data.get("source_image", "unknown"),
        "aruco_pose": {
            "median_rvec": median_rvec,
            "median_tvec": median_tvec
        }
    }
    
    # Save to file
    output_file = "camera_calibration.json"
    with open(output_file, 'w') as f:
        json.dump(visualizer_calibration, f, indent=2)
    
    print(f"Visualizer calibration saved to: {output_file}")
    return True

if __name__ == "__main__":
    if create_visualizer_calibration():
        print("Calibration conversion completed successfully!")
    else:
        print("Calibration conversion failed!")
        exit(1) 