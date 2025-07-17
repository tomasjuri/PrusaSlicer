#!/usr/bin/env python3
"""
Two-step camera calibration routine:
1. Chessboard calibration for camera intrinsics
2. ArUco marker pose estimation for camera extrinsics
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def chessboard_calibration(chessboard_dir, chessboard_size=(9, 6), square_size=0.025):
    """
    Step 1: Chessboard calibration for camera intrinsics
    
    Args:
        chessboard_dir: Directory containing chessboard images
        chessboard_size: Number of internal corners (width, height)
        square_size: Size of chessboard squares in meters
    
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        calibration_error: Reprojection error
    """
    print("Step 1: Chessboard Calibration")
    print("=" * 40)
    
    # Ensure output directories exist
    os.makedirs("calibration/out/imgs", exist_ok=True)
    
    # Check if intrinsics already exist
    intrinsics_file = "calibration/out/camera_intrinsics.json"
    if os.path.exists(intrinsics_file):
        print(f"Loading existing camera intrinsics from {intrinsics_file}")
        with open(intrinsics_file, 'r') as f:
            data = json.load(f)
            camera_matrix = np.array(data['camera_matrix'])
            dist_coeffs = np.array(data['dist_coeffs'])
            calibration_error = data['calibration_error']
            print(f"Loaded intrinsics with error: {calibration_error:.4f}")
            return camera_matrix, dist_coeffs, calibration_error
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get chessboard images
    chessboard_path = Path(chessboard_dir)
    if not chessboard_path.exists():
        raise FileNotFoundError(f"Chessboard directory not found: {chessboard_dir}")
    
    image_files = list(chessboard_path.glob("*.jpg"))
    print(f"Found {len(image_files)} chessboard images")
    
    successful_images = 0
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to load image")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images += 1
            
            # Draw corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            debug_path = f"calibration/out/imgs/debug_chessboard_{img_path.name}"
            cv2.imwrite(debug_path, img)
            print(f"  Found corners, saved debug image")
        else:
            print(f"  No corners found")
    
    if successful_images < 3:
        raise ValueError(f"Need at least 3 successful images, got {successful_images}")
    
    print(f"Successfully processed {successful_images} images")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    calibration_error = mean_error / len(objpoints)
    print(f"Calibration error: {calibration_error:.4f}")
    
    # Save intrinsics
    intrinsics_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'calibration_error': float(calibration_error),
        'image_size': gray.shape[::-1],
        'chessboard_size': chessboard_size,
        'square_size': square_size,
        'successful_images': successful_images
    }
    
    with open(intrinsics_file, 'w') as f:
        json.dump(intrinsics_data, f, indent=2)
    
    print(f"Saved camera intrinsics to {intrinsics_file}")
    
    return camera_matrix, dist_coeffs, calibration_error

def aruco_pose_estimation(aruco_dir, camera_matrix, dist_coeffs, marker_size=0.04, marker_positions_file="calibration/data/aruco_positions.json"):
    """
    Step 2: ArUco marker pose estimation for camera extrinsics
    - Image undistortion
    - Subpixel corner refinement
    - Multi-marker simultaneous pose estimation
    
    Args:
        aruco_dir: Directory containing ArUco marker images
        camera_matrix: Camera intrinsic matrix from chessboard calibration
        dist_coeffs: Distortion coefficients from chessboard calibration
        marker_size: Size of ArUco markers in meters
        marker_positions_file: Path to JSON file containing known marker positions in printer coordinates
    
    Returns:
        poses: List of camera poses (rotation vectors, translation vectors)
        marker_ids: List of detected marker IDs
        image_poses: Dictionary of poses per image
        pose_errors: Dictionary of pose errors per image
    """
    print("\nStep 2: ArUco Pose Estimation")
    print("=" * 40)
    
    # Ensure output directories exist
    os.makedirs("calibration/out/imgs", exist_ok=True)
    
    # Create ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Get ArUco images
    aruco_path = Path(aruco_dir)
    if not aruco_path.exists():
        raise FileNotFoundError(f"ArUco directory not found: {aruco_dir}")
    
    image_files = list(aruco_path.glob("*.jpg"))
    print(f"Found {len(image_files)} ArUco images")
    
    # Load marker positions (required)
    if not os.path.exists(marker_positions_file):
        raise FileNotFoundError(f"Marker positions file not found: {marker_positions_file}")
    
    print(f"Loading marker positions from {marker_positions_file}")
    with open(marker_positions_file, 'r') as f:
        marker_positions_data = json.load(f)
        marker_positions = {}
        for marker_id, pos in marker_positions_data.items():
            # Skip comment fields
            if marker_id.startswith('_'):
                continue
            # Positions are already in meters
            marker_positions[int(marker_id)] = pos
        
    print(f"Loaded {len(marker_positions)} marker positions:")
    for marker_id, pos in marker_positions.items():
        print(f"  ID {marker_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
    
    poses = []
    marker_ids = []
    image_poses = {}
    pose_errors = {}
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to load image")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Undistort the image
        undistorted_gray = cv2.undistort(gray, camera_matrix, dist_coeffs)
        
        # Invert for white printed markers
        inverted_gray = cv2.bitwise_not(undistorted_gray)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(inverted_gray)
        
        if ids is not None:
            print(f"  Detected {len(ids)} markers: {[id[0] for id in ids]}")
            
            # Apply subpixel corner refinement
            refined_corners = apply_subpixel_refinement(undistorted_gray, corners)
            
            # Multi-marker simultaneous pose estimation
            rvec, tvec, success = multi_marker_pose_estimation(
                refined_corners, ids, camera_matrix, dist_coeffs, 
                marker_size, marker_positions
            )
            
            if success:
                # Store individual marker poses
                for i, marker_id in enumerate(ids):
                    poses.append((rvec, tvec))
                    marker_ids.append(marker_id[0])
                
                # Calculate reprojection error for this pose
                # Collect object and image points for reprojection error calculation
                img_object_points = []
                img_image_points = []
                
                for i, marker_id in enumerate(ids):
                    marker_id = marker_id[0]
                    
                    if marker_id in marker_positions:
                        # Use known marker position in printer coordinates - transform to ArUco coordinates
                        marker_center_printer = np.array(marker_positions[marker_id])
                        
                        # Transform from printer coordinates to ArUco coordinates
                        transform_matrix = np.array([
                            [1,  0,  0],  # X stays the same
                            [0, -1,  0],  # Y inverted: printer back -> ArUco down
                            [0,  0, -1]   # Z inverted: printer up -> ArUco forward
                        ])
                        marker_center = transform_matrix @ marker_center_printer
                        
                        half_size = marker_size / 2
                        marker_corners_3d = marker_center + np.array([
                            [-half_size, -half_size, 0],
                            [ half_size, -half_size, 0],
                            [ half_size,  half_size, 0],
                            [-half_size,  half_size, 0]
                        ])
                    else:
                        print(f"WARNING: Marker ID {marker_id} not found in marker_positions!")
                        # Use standard marker coordinate system as fallback
                        marker_corners_3d = marker_size * np.array([
                            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
                        ])
                    
                    img_object_points.append(marker_corners_3d)
                    img_image_points.append(refined_corners[i][0])
                
                if img_object_points:
                    projected_points, _ = cv2.projectPoints(
                        np.vstack(img_object_points), rvec, tvec, camera_matrix, dist_coeffs
                    )
                    projected_points = projected_points.reshape(-1, 2)
                    observed_points = np.vstack(img_image_points)
                    reprojection_error = np.mean(np.linalg.norm(projected_points - observed_points, axis=1))
                    
                    print(f"  Multi-marker pose estimated, reprojection error: {reprojection_error:.2f} pixels")
                    
                    # Store pose data
                    image_poses[img_path.name] = {
                        'rvec': rvec.flatten().tolist(),
                        'tvec': tvec.flatten().tolist(),
                        'num_markers': len(ids)
                    }
                    pose_errors[img_path.name] = {
                        'reprojection_error': reprojection_error,
                        'num_markers': len(ids)
                    }
            else:
                print(f"  Failed to estimate pose for this image")
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(img, refined_corners, ids)
            
            # Draw individual marker axes
            if success and rvec is not None and tvec is not None:
                draw_individual_marker_axes(img, refined_corners, ids, camera_matrix, dist_coeffs, 
                                          marker_size, marker_positions, rvec, tvec)
            
                            # Draw global printer coordinate system axes at origin
                if success and rvec is not None and tvec is not None:
                    draw_global_printer_axes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 3)
                    
                    # Add camera pose information in ArUco coordinates
                    add_camera_pose_info(img, rvec, tvec)
                
                # Add text with pose information
                cv2.putText(img, f"Multi-Marker Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Markers: {len(ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if img_object_points:
                    cv2.putText(img, f"Reproj Error: {reprojection_error:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add coordinate system labels
                cv2.putText(img, f"Global Printer Coordinates:", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img, f"Red: X-axis (right), Green: Y-axis (back), Blue: Z-axis (up)", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
            # Save debug image
            debug_path = f"calibration/out/imgs/debug_aruco_{img_path.name}"
            cv2.imwrite(debug_path, img)
            print(f"  Saved debug image with pose visualization to {debug_path}")
        else:
            print(f"  No markers detected")
    

    
    print(f"\nTotal poses estimated: {len(poses)}")
    print(f"Detected marker IDs: {sorted(set(marker_ids))}")
    
    return poses, marker_ids, image_poses, pose_errors

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_calibration_results(camera_matrix, dist_coeffs, poses, marker_ids, image_poses, pose_errors, output_file="calibration/out/camera_calibration.json"):
    """Save calibration results to JSON file"""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert poses to lists for JSON serialization
    poses_list = []
    for rvec, tvec in poses:
        poses_list.append({
            'rotation_vector': rvec.flatten().tolist(),
            'translation_vector': tvec.flatten().tolist()
        })
    
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'poses': poses_list,
        'marker_ids': [int(id) for id in marker_ids],  # Convert numpy int32 to regular int
        'image_poses': convert_numpy_types(image_poses),
        'pose_errors': convert_numpy_types(pose_errors),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration results saved to {output_file}")

def apply_subpixel_refinement(gray, corners, pattern_size=None):
    """
    Apply subpixel corner refinement for improved accuracy
    
    Args:
        gray: Grayscale image
        corners: Detected corners
        pattern_size: For chessboard, use pattern size. For ArUco, use None
    
    Returns:
        refined_corners: Subpixel-refined corners
    """
    if corners is None or len(corners) == 0:
        return corners
    
    # Criteria for subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    if pattern_size is not None:
        # Chessboard corners
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        # ArUco marker corners - refine each marker's corners
        refined_corners = []
        for corner_set in corners:
            refined_corner_set = cv2.cornerSubPix(gray, corner_set, (5, 5), (-1, -1), criteria)
            refined_corners.append(refined_corner_set)
    
    return refined_corners

def multi_marker_pose_estimation(corners, ids, camera_matrix, dist_coeffs, marker_size=0.04, marker_positions=None):
    """
    Estimate camera pose using all detected markers simultaneously
    
    Args:
        corners: List of marker corner arrays
        ids: List of marker IDs
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_size: Size of markers in meters
        marker_positions: Dictionary of known marker positions in printer coordinates
    
    Returns:
        rvec: Camera rotation vector
        tvec: Camera translation vector
        success: Whether pose estimation succeeded
    """
    if len(corners) == 0:
        return None, None, False
    
    # Prepare 3D object points and 2D image points
    all_object_points = []
    all_image_points = []
    
    for i, marker_id in enumerate(ids):
        marker_id = marker_id[0]  # Extract ID from array
        
        if marker_positions is not None and marker_id in marker_positions:
            # Use known marker position in printer coordinates - transform to ArUco coordinates
            marker_center_printer = np.array(marker_positions[marker_id])
            
            # Transform from printer coordinates to ArUco coordinates
            transform_matrix = np.array([
                [1,  0,  0],  # X stays the same
                [0, -1,  0],  # Y inverted: printer back -> ArUco down
                [0,  0, -1]   # Z inverted: printer up -> ArUco forward
            ])
            marker_center = transform_matrix @ marker_center_printer
            
            # Define marker corners relative to center (40mm square)
            half_size = marker_size / 2
            marker_corners_3d = marker_center + np.array([
                [-half_size, -half_size, 0],  # Bottom-left
                [ half_size, -half_size, 0],  # Bottom-right
                [ half_size,  half_size, 0],  # Top-right
                [-half_size,  half_size, 0]   # Top-left
            ])
        else:
            # Fall back to standard marker coordinate system (marker at origin)
            print(f"WARNING: Marker ID {marker_id} not found in marker_positions!")
            marker_corners_3d = marker_size * np.array([
                [0, 0, 0],  # Bottom-left
                [1, 0, 0],  # Bottom-right
                [1, 1, 0],  # Top-right
                [0, 1, 0]   # Top-left
            ])
        
        all_object_points.append(marker_corners_3d)
        all_image_points.append(corners[i][0])  # corners[i] is shape (1, 4, 2)
    
    # Combine all points
    object_points = np.vstack(all_object_points).astype(np.float32)
    image_points = np.vstack(all_image_points).astype(np.float32)
    
    # Estimate pose using all markers simultaneously
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return rvec, tvec, success

def draw_individual_marker_axes(img, corners, ids, camera_matrix, dist_coeffs, 
                               marker_size, marker_positions, global_rvec, global_tvec):
    """
    Draw coordinate axes for each individual ArUco marker
    
    Args:
        img: Image to draw on
        corners: List of marker corner arrays
        ids: List of marker IDs
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_size: Size of markers in meters
        marker_positions: Dictionary of known marker positions in printer coordinates
        global_rvec: Global rotation vector
        global_tvec: Global translation vector
    """
    if len(corners) == 0 or ids is None or len(ids) == 0:
        return
    
    axes_length = marker_size * 0.8  # Slightly smaller than marker size
    
    for i, marker_id in enumerate(ids):
        marker_id = marker_id[0]
        
        if marker_id in marker_positions:
            # Get marker center in printer coordinates - transform to ArUco coordinates
            marker_center_printer = np.array(marker_positions[marker_id])
            
            # Transform from printer coordinates to ArUco coordinates
            transform_matrix = np.array([
                [1,  0,  0],  # X stays the same
                [0, -1,  0],  # Y inverted: printer back -> ArUco down
                [0,  0, -1]   # Z inverted: printer up -> ArUco forward
            ])
            marker_center = transform_matrix @ marker_center_printer
            
            # Define axes points in marker's local coordinate system
            marker_axes_3d = np.array([
                marker_center,                                                # Origin
                marker_center + [axes_length, 0, 0],                        # X-axis (red)
                marker_center,                                                # Origin
                marker_center + [0, axes_length, 0],                        # Y-axis (green)
                marker_center,                                                # Origin
                marker_center + [0, 0, axes_length]                         # Z-axis (blue)
            ], dtype=np.float32)
            
            # Project to image coordinates
            marker_axes_projected, _ = cv2.projectPoints(
                marker_axes_3d, global_rvec, global_tvec, camera_matrix, dist_coeffs
            )
            marker_axes_projected = marker_axes_projected.reshape(-1, 2).astype(int)
            
            # Draw marker axes (thinner lines than global axes)
            cv2.line(img, tuple(marker_axes_projected[0]), tuple(marker_axes_projected[1]), (0, 0, 200), 2)    # X-axis: Dark Red
            cv2.line(img, tuple(marker_axes_projected[2]), tuple(marker_axes_projected[3]), (0, 200, 0), 2)    # Y-axis: Dark Green
            cv2.line(img, tuple(marker_axes_projected[4]), tuple(marker_axes_projected[5]), (200, 0, 0), 2)    # Z-axis: Dark Blue
            
            # Add marker ID label near the origin
            label_pos = tuple(marker_axes_projected[0] + [5, -5])
            cv2.putText(img, f"ID{marker_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_global_printer_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axes_length):
    """
    Draw global printer coordinate system axes at the printer origin (0, 0, 0)
    If origin is outside image bounds, place axes at printer board center instead
    
    Args:
        img: Image to draw on
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Camera rotation vector
        tvec: Camera translation vector
        axes_length: Length of the axes in meters
    """
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Printer coordinate origin (0, 0, 0) in printer coordinates
    printer_origin = np.array([0.0, 0.0, 0.0])
    
    # Transform from printer coordinates to ArUco coordinates
    transform_matrix = np.array([
        [1,  0,  0],  # X stays the same
        [0, -1,  0],  # Y inverted: printer back -> ArUco down
        [0,  0, -1]   # Z inverted: printer up -> ArUco forward
    ])
    
    # Transform origin to ArUco coordinates and project to image
    aruco_origin = transform_matrix @ printer_origin
    origin_projected, _ = cv2.projectPoints(
        aruco_origin.reshape(1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    origin_2d = origin_projected.reshape(2).astype(int)
    
    # Check if origin is within image bounds (with some margin)
    margin = 50
    origin_in_bounds = (margin <= origin_2d[0] <= img_width - margin and 
                       margin <= origin_2d[1] <= img_height - margin)
    
    # If origin is out of bounds, use printer board center instead
    if not origin_in_bounds:
        # Calculate printer board center based on ArUco marker positions
        # Markers range from (0.04, 0.04) to (0.22, 0.17) in printer coordinates
        printer_board_center = np.array([0.13, 0.105, 0.0])  # Center of printer board
        printer_origin = printer_board_center
        label_text = "BOARD CENTER"
        label_color = (255, 255, 0)  # Yellow for board center
    else:
        label_text = "ORIGIN"
        label_color = (255, 255, 255)  # White for true origin
    
    # Define printer coordinate axes directions in printer coordinates
    printer_x_axis = np.array([axes_length, 0.0, 0.0])  # X: right
    printer_y_axis = np.array([0.0, axes_length, 0.0])  # Y: back  
    printer_z_axis = np.array([0.0, 0.0, axes_length])  # Z: up
    
    # Transform origin and axes to ArUco coordinates
    aruco_origin = transform_matrix @ printer_origin
    aruco_x_end = transform_matrix @ (printer_origin + printer_x_axis)
    aruco_y_end = transform_matrix @ (printer_origin + printer_y_axis)
    aruco_z_end = transform_matrix @ (printer_origin + printer_z_axis)
    
    # Combine points for projection
    axes_points_3d = np.array([
        aruco_origin,   # Origin
        aruco_x_end,    # X-axis end
        aruco_y_end,    # Y-axis end
        aruco_z_end     # Z-axis end
    ], dtype=np.float32)
    
    # Project to image coordinates
    axes_projected, _ = cv2.projectPoints(
        axes_points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    axes_projected = axes_projected.reshape(-1, 2).astype(int)
    
    origin_2d = tuple(axes_projected[0])
    x_end_2d = tuple(axes_projected[1])
    y_end_2d = tuple(axes_projected[2])
    z_end_2d = tuple(axes_projected[3])
    
    # Draw axes with printer coordinate system colors and labels
    # X-axis: Red (printer right)
    cv2.line(img, origin_2d, x_end_2d, (0, 0, 255), 6)
    cv2.putText(img, "X", (x_end_2d[0] + 10, x_end_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Y-axis: Green (printer back)
    cv2.line(img, origin_2d, y_end_2d, (0, 255, 0), 6)
    cv2.putText(img, "Y", (y_end_2d[0] + 10, y_end_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Z-axis: Blue (printer up)
    cv2.line(img, origin_2d, z_end_2d, (255, 0, 0), 6)
    cv2.putText(img, "Z", (z_end_2d[0] + 10, z_end_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # Draw origin marker
    cv2.circle(img, origin_2d, 8, (255, 255, 255), -1)
    cv2.circle(img, origin_2d, 8, (0, 0, 0), 2)
    cv2.putText(img, label_text, (origin_2d[0] + 15, origin_2d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

def add_camera_pose_info(img, rvec, tvec, scale=1.0):
    """
    Add camera position and orientation information in both ArUco and Printer coordinate systems
    
    Args:
        img: Image to draw on
        rvec: Camera rotation vector (in ArUco coordinates)
        tvec: Camera translation vector (in ArUco coordinates)
    """
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Camera position in ArUco coordinates (inverse transformation)
    # Camera pose gives object->camera transformation, we need camera->object
    camera_pos_aruco = -rmat.T @ tvec.flatten()
    
    # Transform camera position from ArUco coordinates to Printer coordinates
    # This is the inverse of the transformation we use for markers
    inverse_transform_matrix = np.array([
        [1,  0,  0],  # X stays the same
        [0, -1,  0],  # Y inverted: ArUco down -> printer back
        [0,  0, -1]   # Z inverted: ArUco forward -> printer up
    ])
    camera_pos_printer = inverse_transform_matrix @ camera_pos_aruco
    
    # Transform rotation matrix to printer coordinates
    # ArUco rotation matrix -> Printer rotation matrix
    transform_matrix = np.array([
        [1,  0,  0],  # X stays the same
        [0, -1,  0],  # Y inverted: ArUco down -> printer back
        [0,  0, -1]   # Z inverted: ArUco forward -> printer up
    ])
    # R_printer = T * R_aruco * T^(-1)
    rmat_printer = transform_matrix @ rmat @ transform_matrix.T
    
    # Extract Euler angles from rotation matrix (in Printer coordinates)
    # Using ZYX convention: Yaw (Z), Pitch (Y), Roll (X)
    sy = np.sqrt(rmat_printer[0,0] * rmat_printer[0,0] + rmat_printer[1,0] * rmat_printer[1,0])
    singular = sy < 1e-6
    
    if not singular:
        # Roll: rotation around X-axis (right) - camera tilt left/right
        roll = np.arctan2(rmat_printer[2,1], rmat_printer[2,2])   
        # Pitch: rotation around Y-axis (back) - camera look up/down  
        pitch = np.arctan2(-rmat_printer[2,0], sy)        
        # Yaw: rotation around Z-axis (up) - camera pan left/right
        yaw = np.arctan2(rmat_printer[1,0], rmat_printer[0,0])    
    else:
        roll = np.arctan2(-rmat_printer[1,2], rmat_printer[1,1])
        pitch = np.arctan2(-rmat_printer[2,0], sy)
        yaw = 0
    
    # Convert to degrees
    angles = np.array([roll, pitch, yaw]) * 180.0 / np.pi
    
    # Create text overlay (right side of image) - larger for dual coordinate display
    text_x = img.shape[1] - 1200
    text_y = 80
    line_height = int(85 * scale)
    col2_offset = 600  # Offset for second column
    
    # Background rectangle for better readability - larger for dual display
    overlay = img.copy()
    cv2.rectangle(overlay, (text_x - 40, text_y - 50), (img.shape[1] - 10, text_y + line_height * 9), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Header
    cv2.putText(img, "CAMERA POSE", (text_x + 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    
    # Column headers
    cv2.putText(img, "ArUco Coords", (text_x, text_y + int(1.5*line_height)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(img, "Printer Coords", (text_x + col2_offset, text_y + int(1.5*line_height)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # Position data - ArUco coordinates
    cv2.putText(img, f"X: {camera_pos_aruco[0]*100:.1f}cm", (text_x, text_y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(img, f"Y: {camera_pos_aruco[1]*100:.1f}cm", (text_x, text_y + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(img, f"Z: {camera_pos_aruco[2]*100:.1f}cm", (text_x, text_y + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    
    # Position data - Printer coordinates
    cv2.putText(img, f"X: {camera_pos_printer[0]*100:.1f}cm", (text_x + col2_offset, text_y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(img, f"Y: {camera_pos_printer[1]*100:.1f}cm", (text_x + col2_offset, text_y + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(img, f"Z: {camera_pos_printer[2]*100:.1f}cm", (text_x + col2_offset, text_y + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    
    # Angles (in printer coordinates - displayed centered)
    angle_x = text_x + 200
    cv2.putText(img, "Orientation (Printer Coords)", (angle_x, text_y + int(6.5*line_height)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 200, 100), 3)
    cv2.putText(img, f"Roll: {angles[0]:.1f}° (tilt L/R)", (angle_x, text_y + 7*line_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, f"Pitch: {angles[1]:.1f}° (look U/D)", (angle_x, text_y + 8*line_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, f"Yaw: {angles[2]:.1f}° (pan L/R)", (angle_x, text_y + 9*line_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

def main():
    parser = argparse.ArgumentParser(description="Two-step camera calibration")
    parser.add_argument("--chessboard-dir", default="calibration/chessboard", 
                       help="Directory containing chessboard images")
    parser.add_argument("--aruco-dir", default="calibration/aruco",
                       help="Directory containing ArUco marker images")
    parser.add_argument("--chessboard-size", nargs=2, type=int, default=[9, 6],
                       help="Chessboard internal corners (width height)")
    parser.add_argument("--square-size", type=float, default=0.020,
                       help="Chessboard square size in meters")
    parser.add_argument("--marker-size", type=float, default=0.04,
                       help="ArUco marker size in meters")
    parser.add_argument("--marker-positions", default="calibration/data/aruco_positions.json",
                       help="JSON file containing known marker positions in world coordinates")
    parser.add_argument("--output", default="calibration/out/camera_calibration.json",
                       help="Output calibration file")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Chessboard calibration
        camera_matrix, dist_coeffs, calibration_error = chessboard_calibration(
            args.chessboard_dir, tuple(args.chessboard_size), args.square_size
        )
        
        # Step 2: ArUco pose estimation
        poses, marker_ids, image_poses, pose_errors = aruco_pose_estimation(
            args.aruco_dir, camera_matrix, dist_coeffs, args.marker_size, args.marker_positions
        )
        
        # Save results
        save_calibration_results(camera_matrix, dist_coeffs, poses, marker_ids, image_poses, pose_errors, args.output)
        
        print("\nCalibration completed successfully!")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.flatten()}")
        print(f"Calibration error: {calibration_error:.4f}")
        print(f"Detected {len(poses)} poses from {len(set(marker_ids))} unique markers")
        
        # Print pose error summary
        if pose_errors:
            print(f"\nPose Error Summary:")
            print(f"  Images with poses: {len(pose_errors)}")
            reprojection_errors = [data['reprojection_error'] for data in pose_errors.values()]
            print(f"  Average reprojection error: {np.mean(reprojection_errors):.2f} pixels")
            print(f"  Min reprojection error: {np.min(reprojection_errors):.2f} pixels")
            print(f"  Max reprojection error: {np.max(reprojection_errors):.2f} pixels")
            print(f"  Best image: {min(pose_errors.items(), key=lambda x: x[1]['reprojection_error'])[0]}")
            print(f"  Worst image: {max(pose_errors.items(), key=lambda x: x[1]['reprojection_error'])[0]}")
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 