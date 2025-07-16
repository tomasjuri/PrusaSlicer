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
from scipy.optimize import least_squares
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

def compute_median_pose(poses):
    """
    Compute median pose from a list of poses
    
    Args:
        poses: List of (rvec, tvec) tuples
    
    Returns:
        median_rvec: Median rotation vector
        median_tvec: Median translation vector
        pose_errors: List of pose errors from median
    """
    if len(poses) == 1:
        return poses[0][0], poses[0][1], [0.0]
    
    # Convert rotation vectors to rotation matrices
    rmatrices = []
    for rvec, _ in poses:
        rmat, _ = cv2.Rodrigues(rvec)
        rmatrices.append(rmat)
    
    # Compute median rotation matrix
    rmatrices_array = np.array(rmatrices)
    median_rmat = np.median(rmatrices_array, axis=0)
    
    # Convert back to rotation vector
    median_rvec, _ = cv2.Rodrigues(median_rmat)
    
    # Compute median translation vector
    tvecs = np.array([tvec.flatten() for _, tvec in poses])
    median_tvec = np.median(tvecs, axis=0).reshape(3, 1)
    
    # Compute pose errors
    pose_errors = []
    for rvec, tvec in poses:
        # Rotation error (angle difference in degrees)
        rmat, _ = cv2.Rodrigues(rvec)
        angle_diff = np.arccos(np.clip((np.trace(rmat.T @ median_rmat) - 1) / 2, -1, 1))
        rotation_error = np.degrees(angle_diff)
        
        # Translation error (Euclidean distance)
        translation_error = np.linalg.norm(tvec.flatten() - median_tvec.flatten())
        
        # Combined error (weighted)
        total_error = rotation_error + translation_error * 1000  # Scale translation error
        pose_errors.append(total_error)
    
    return median_rvec, median_tvec, pose_errors

def aruco_pose_estimation(aruco_dir, camera_matrix, dist_coeffs, marker_size=0.04, marker_positions_file="calibration/data/aruco_positions.json"):
    """
    Step 2: ArUco marker pose estimation for camera extrinsics with improvements:
    - Image undistortion
    - Subpixel corner refinement
    - Multi-marker simultaneous pose estimation
    - Bundle adjustment optimization
    
    Args:
        aruco_dir: Directory containing ArUco marker images
        camera_matrix: Camera intrinsic matrix from chessboard calibration
        dist_coeffs: Distortion coefficients from chessboard calibration
        marker_size: Size of ArUco markers in meters
        marker_positions_file: Path to JSON file containing known marker positions in world coordinates
    
    Returns:
        poses: List of camera poses (rotation vectors, translation vectors)
        marker_ids: List of detected marker IDs
        image_poses: Dictionary of optimized poses per image
        pose_errors: Dictionary of pose errors per image
    """
    print("\nStep 2: Enhanced ArUco Pose Estimation")
    print("=" * 50)
    
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
    
    # Data for bundle adjustment
    all_rvecs = []
    all_tvecs = []
    all_object_points = []
    all_image_points = []
    all_camera_matrices = []
    all_dist_coeffs = []
    valid_images = []
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to load image")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # IMPROVEMENT 1: Undistort the image
        undistorted_gray = cv2.undistort(gray, camera_matrix, dist_coeffs)
        
        # Invert for white printed markers
        inverted_gray = cv2.bitwise_not(undistorted_gray)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(inverted_gray)
        
        if ids is not None:
            print(f"  Detected {len(ids)} markers: {[id[0] for id in ids]}")
            
            # IMPROVEMENT 2: Apply subpixel corner refinement
            refined_corners = apply_subpixel_refinement(undistorted_gray, corners)
            
            # IMPROVEMENT 3: Multi-marker simultaneous pose estimation
            rvec, tvec, success = multi_marker_pose_estimation(
                refined_corners, ids, camera_matrix, dist_coeffs, 
                marker_size, marker_positions, coordinate_transform=True
            )
            
            if success:
                # Store data for bundle adjustment
                all_rvecs.append(rvec)
                all_tvecs.append(tvec)
                all_camera_matrices.append(camera_matrix)
                all_dist_coeffs.append(dist_coeffs)
                valid_images.append(img_path.name)
                
                # Store object points and image points for bundle adjustment
                img_object_points = []
                img_image_points = []
                
                for i, marker_id in enumerate(ids):
                    marker_id = marker_id[0]
                    
                    if marker_id in marker_positions:
                        # Use known marker position in world coordinates
                        marker_center = np.array(marker_positions[marker_id])
                        
                        # Apply coordinate transformation (printer -> OpenCV coordinates)
                        transform_matrix = np.array([
                            [1,  0,  0],  # X stays the same
                            [0, -1,  0],  # Y inverted 
                            [0,  0, -1]   # Z inverted
                        ])
                        marker_center = transform_matrix @ marker_center
                        
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
                
                # Store combined points for this image
                if img_object_points:
                    all_object_points.append(np.vstack(img_object_points).astype(np.float32))
                    all_image_points.append(np.vstack(img_image_points).astype(np.float32))
                else:
                    all_object_points.append(np.array([]).reshape(0, 3))
                    all_image_points.append(np.array([]).reshape(0, 2))
                
                # Store individual marker poses for compatibility
                for i, marker_id in enumerate(ids):
                    poses.append((rvec, tvec))
                    marker_ids.append(marker_id[0])
                
                # Calculate reprojection error for this pose
                if len(img_object_points) > 0:
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
                    all_object_points.append(np.array([]).reshape(0, 3))
                    all_image_points.append(np.array([]).reshape(0, 2))
            else:
                print(f"  Failed to estimate pose for this image")
                all_rvecs.append(None)
                all_tvecs.append(None)
                all_camera_matrices.append(camera_matrix)
                all_dist_coeffs.append(dist_coeffs)
                all_object_points.append(np.array([]).reshape(0, 3))
                all_image_points.append(np.array([]).reshape(0, 2))
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(img, refined_corners, ids)
            
            # Draw pose axes and printer bed if successful
            if success and rvec is not None and tvec is not None:
                # Draw coordinate axes for the multi-marker pose
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 2)
                
                # Draw printer bed outline based on marker positions
                # Markers are centered at their positions with 40mm size
                # So bed starts 20mm before first marker and ends 20mm after last marker
                bed_corners_3d = np.array([
                    [0.020, 0.020, 0.0],  # Start 20mm before first marker (40-20, 40-20)
                    [0.240, 0.020, 0.0],  # End 20mm after last marker (220+20, 40-20)
                    [0.240, 0.190, 0.0],  # Far corner (220+20, 170+20)
                    [0.020, 0.190, 0.0],  # Y-axis corner (40-20, 170+20)
                    [0.020, 0.020, 0.0]   # Back to origin
                ], dtype=np.float32)
                
                # Project bed corners to image
                bed_projected, _ = cv2.projectPoints(bed_corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
                bed_projected = bed_projected.reshape(-1, 2).astype(int)
                
                # Draw bed outline
                for i in range(len(bed_projected) - 1):
                    cv2.line(img, tuple(bed_projected[i]), tuple(bed_projected[i + 1]), (255, 255, 0), 3)  # Cyan bed outline
                
                # Draw printer coordinate axes at bed origin (20, 20)
                bed_origin = [0.020, 0.020, 0.0]
                printer_axes_3d = np.array([
                    bed_origin,                           # Bed origin
                    [bed_origin[0] + 0.05, bed_origin[1], bed_origin[2]],  # X-axis (50mm red)
                    bed_origin,                           # Bed origin
                    [bed_origin[0], bed_origin[1] + 0.05, bed_origin[2]],  # Y-axis (50mm green)
                    bed_origin,                           # Bed origin
                    [bed_origin[0], bed_origin[1], bed_origin[2] + 0.05]   # Z-axis (50mm blue)
                ], dtype=np.float32)
                
                printer_axes_projected, _ = cv2.projectPoints(printer_axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
                printer_axes_projected = printer_axes_projected.reshape(-1, 2).astype(int)
                
                # Draw printer axes (thicker lines)
                cv2.line(img, tuple(printer_axes_projected[0]), tuple(printer_axes_projected[1]), (0, 0, 255), 4)    # X-axis: Red
                cv2.line(img, tuple(printer_axes_projected[2]), tuple(printer_axes_projected[3]), (0, 255, 0), 4)    # Y-axis: Green
                cv2.line(img, tuple(printer_axes_projected[4]), tuple(printer_axes_projected[5]), (255, 0, 0), 4)    # Z-axis: Blue
                
                # Add text with pose information
                cv2.putText(img, f"Multi-Marker Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Markers: {len(ids)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if len(img_object_points) > 0:
                    cv2.putText(img, f"Reproj Error: {reprojection_error:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add coordinate system labels
                cv2.putText(img, f"Cyan: Bed Outline", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, f"Red: X-axis, Green: Y-axis, Blue: Z-axis", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save debug image
            debug_path = f"calibration/out/imgs/debug_aruco_{img_path.name}"
            cv2.imwrite(debug_path, img)
            print(f"  Saved debug image with pose visualization")
        else:
            print(f"  No markers detected")
            all_rvecs.append(None)
            all_tvecs.append(None)
            all_camera_matrices.append(camera_matrix)
            all_dist_coeffs.append(dist_coeffs)
            all_object_points.append(np.array([]).reshape(0, 3))
            all_image_points.append(np.array([]).reshape(0, 2))
    
    # IMPROVEMENT 4: Bundle Adjustment
    print(f"\nStep 3: Bundle Adjustment Optimization")
    print("=" * 50)
    
    valid_indices = [i for i, rvec in enumerate(all_rvecs) if rvec is not None]
    print(f"Performing bundle adjustment on {len(valid_indices)} valid poses...")
    
    if len(valid_indices) >= 2:
        # Filter to only valid poses
        valid_rvecs = [all_rvecs[i] for i in valid_indices]
        valid_tvecs = [all_tvecs[i] for i in valid_indices]
        valid_cameras = [all_camera_matrices[i] for i in valid_indices]
        valid_dist_coeffs = [all_dist_coeffs[i] for i in valid_indices]
        valid_object_points = [all_object_points[i] for i in valid_indices]
        valid_image_points = [all_image_points[i] for i in valid_indices]
        
        # Perform bundle adjustment
        optimized_rvecs, optimized_tvecs, optimization_result = bundle_adjustment(
            valid_rvecs, valid_tvecs, valid_cameras, valid_dist_coeffs,
            valid_object_points, valid_image_points
        )
        
        # Update the poses with optimized values
        opt_idx = 0
        for i, valid_idx in enumerate(valid_indices):
            if optimized_rvecs[opt_idx] is not None:
                # Update image poses with optimized values
                img_name = valid_images[valid_idx]
                if img_name in image_poses:
                    image_poses[img_name]['rvec'] = optimized_rvecs[opt_idx].flatten().tolist()
                    image_poses[img_name]['tvec'] = optimized_tvecs[opt_idx].flatten().tolist()
                    
                    # Recalculate reprojection error with optimized pose
                    if len(valid_object_points[opt_idx]) > 0:
                        projected_points, _ = cv2.projectPoints(
                            valid_object_points[opt_idx], optimized_rvecs[opt_idx], optimized_tvecs[opt_idx],
                            valid_cameras[opt_idx], valid_dist_coeffs[opt_idx]
                        )
                        projected_points = projected_points.reshape(-1, 2)
                        observed_points = valid_image_points[opt_idx]
                        optimized_error = np.mean(np.linalg.norm(projected_points - observed_points, axis=1))
                        
                        pose_errors[img_name]['optimized_reprojection_error'] = optimized_error
                        print(f"  {img_name}: {pose_errors[img_name]['reprojection_error']:.2f} -> {optimized_error:.2f} pixels")
            
            opt_idx += 1
        
        print(f"Bundle adjustment optimization completed!")
        print(f"Final optimization cost: {optimization_result.cost:.6f}")
    else:
        print(f"Not enough valid poses for bundle adjustment (need ≥2, got {len(valid_indices)})")
    
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

def multi_marker_pose_estimation(corners, ids, camera_matrix, dist_coeffs, marker_size=0.04, marker_positions=None, coordinate_transform=True):
    """
    Estimate camera pose using all detected markers simultaneously
    
    Args:
        corners: List of marker corner arrays
        ids: List of marker IDs
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_size: Size of markers in meters
        marker_positions: Dictionary of known marker positions in world coordinates
    
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
            # Use known marker position in world coordinates
            marker_center = np.array(marker_positions[marker_id])
            
            # Apply coordinate transformation if needed (printer -> OpenCV coordinates)
            if coordinate_transform:
                # Transform from printer coordinates (X=right, Y=back, Z=up) 
                # to OpenCV coordinates (X=right, Y=down, Z=forward)
                # This matches the transformation we used in visualization
                transform_matrix = np.array([
                    [1,  0,  0],  # X stays the same
                    [0, -1,  0],  # Y inverted 
                    [0,  0, -1]   # Z inverted
                ])
                marker_center = transform_matrix @ marker_center
            
            # Define marker corners relative to center (40mm square)
            half_size = marker_size / 2
            marker_corners_3d = marker_center + np.array([
                [-half_size, -half_size, 0],  # Bottom-left
                [ half_size, -half_size, 0],  # Bottom-right
                [ half_size,  half_size, 0],  # Top-right
                [-half_size,  half_size, 0]   # Top-left
            ])
        else:
            # This should not happen if marker_positions is properly provided
            print(f"WARNING: Marker ID {marker_id} not found in marker_positions!")
            # Fall back to standard marker coordinate system (marker at origin)
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

def bundle_adjustment_objective(params, camera_matrices, dist_coeffs, all_object_points, all_image_points, num_images):
    """
    Objective function for bundle adjustment
    
    Args:
        params: Flattened parameters [rvecs, tvecs, object_points_refinement]
        camera_matrices: List of camera matrices (assuming fixed intrinsics)
        dist_coeffs: List of distortion coefficients
        all_object_points: List of 3D object points for each image
        all_image_points: List of 2D image points for each image
        num_images: Number of images
    
    Returns:
        residuals: Reprojection errors flattened
    """
    # Parse parameters
    param_idx = 0
    
    # Extract camera poses (6 parameters per image: 3 for rotation, 3 for translation)
    rvecs = []
    tvecs = []
    for i in range(num_images):
        rvec = params[param_idx:param_idx+3].reshape(3, 1)
        tvec = params[param_idx+3:param_idx+6].reshape(3, 1)
        rvecs.append(rvec)
        tvecs.append(tvec)
        param_idx += 6
    
    # For this implementation, we'll keep object points fixed
    # In full bundle adjustment, you would also optimize 3D point positions
    
    residuals = []
    
    for i in range(num_images):
        if len(all_object_points[i]) == 0:
            continue
            
        # Project 3D points to image
        projected_points, _ = cv2.projectPoints(
            all_object_points[i], rvecs[i], tvecs[i], 
            camera_matrices[i], dist_coeffs[i]
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Compute residuals
        image_points = all_image_points[i].reshape(-1, 2)
        residual = (projected_points - image_points).flatten()
        residuals.extend(residual)
    
    return np.array(residuals)

def bundle_adjustment(initial_rvecs, initial_tvecs, camera_matrices, dist_coeffs, 
                     all_object_points, all_image_points):
    """
    Perform bundle adjustment to refine camera poses
    
    Args:
        initial_rvecs: Initial rotation vectors for each image
        initial_tvecs: Initial translation vectors for each image
        camera_matrices: Camera matrices for each image
        dist_coeffs: Distortion coefficients for each image
        all_object_points: 3D object points for each image
        all_image_points: 2D image points for each image
    
    Returns:
        optimized_rvecs: Refined rotation vectors
        optimized_tvecs: Refined translation vectors
        optimization_result: scipy optimization result
    """
    num_images = len(initial_rvecs)
    
    # Prepare initial parameters
    initial_params = []
    for i in range(num_images):
        if initial_rvecs[i] is not None and initial_tvecs[i] is not None:
            initial_params.extend(initial_rvecs[i].flatten())
            initial_params.extend(initial_tvecs[i].flatten())
        else:
            # Use zero pose if no initial estimate
            initial_params.extend([0, 0, 0, 0, 0, 0])
    
    initial_params = np.array(initial_params)
    
    print(f"Starting bundle adjustment with {num_images} images...")
    print(f"Initial parameter vector size: {len(initial_params)}")
    
    # Run optimization
    result = least_squares(
        bundle_adjustment_objective,
        initial_params,
        args=(camera_matrices, dist_coeffs, all_object_points, all_image_points, num_images),
        method='lm',  # Levenberg-Marquardt
        verbose=1
    )
    
    # Parse optimized parameters
    param_idx = 0
    optimized_rvecs = []
    optimized_tvecs = []
    
    for i in range(num_images):
        if initial_rvecs[i] is not None:
            rvec = result.x[param_idx:param_idx+3].reshape(3, 1)
            tvec = result.x[param_idx+3:param_idx+6].reshape(3, 1)
        else:
            rvec = None
            tvec = None
        
        optimized_rvecs.append(rvec)
        optimized_tvecs.append(tvec)
        param_idx += 6
    
    print(f"Bundle adjustment completed. Final cost: {result.cost:.6f}")
    
    return optimized_rvecs, optimized_tvecs, result

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