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

def aruco_pose_estimation(aruco_dir, camera_matrix, dist_coeffs, marker_size=0.04, marker_positions=None):
    """
    Step 2: ArUco marker pose estimation for camera extrinsics
    
    Args:
        aruco_dir: Directory containing ArUco marker images
        camera_matrix: Camera intrinsic matrix from chessboard calibration
        dist_coeffs: Distortion coefficients from chessboard calibration
        marker_size: Size of ArUco markers in meters
        marker_positions: Optional list of known marker positions in world coordinates
    
    Returns:
        poses: List of camera poses (rotation vectors, translation vectors)
        marker_ids: List of detected marker IDs
        image_poses: Dictionary of median poses per image
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
        
        # Convert to grayscale and INVERT for white printed markers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted_gray = cv2.bitwise_not(gray)  # Invert for white printed markers
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(inverted_gray)
        
        if ids is not None:
            print(f"  Detected {len(ids)} markers: {[id[0] for id in ids]}")
            
            # Collect poses for this image
            image_poses_list = []
            
            # Estimate pose for each marker
            for i, marker_id in enumerate(ids):
                # Estimate pose
                ret, rvec, tvec = cv2.solvePnP(
                    marker_size * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                    corners[i][0],
                    camera_matrix,
                    dist_coeffs
                )
                
                poses.append((rvec, tvec))
                marker_ids.append(marker_id[0])
                image_poses_list.append((rvec, tvec))
            
            # Compute median pose for this image
            if len(image_poses_list) > 0:
                median_rvec, median_tvec, errors = compute_median_pose(image_poses_list)
                image_poses[img_path.name] = {
                    'median_rvec': median_rvec.flatten().tolist(),
                    'median_tvec': median_tvec.flatten().tolist(),
                    'num_markers': len(image_poses_list)
                }
                pose_errors[img_path.name] = {
                    'errors': errors,
                    'mean_error': np.mean(errors),
                    'max_error': np.max(errors),
                    'std_error': np.std(errors)
                }
                
                print(f"  Median pose computed, mean error: {np.mean(errors):.2f}°")
                
                # Check for high errors
                if np.max(errors) > 10.0:  # 10 degrees threshold
                    print(f"  WARNING: High pose variation detected! Max error: {np.max(errors):.2f}°")
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # Draw median pose axes
            if len(image_poses_list) > 0:
                # Draw coordinate axes for median pose
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, median_rvec, median_tvec, marker_size * 2)
                
                # Add text with pose information
                cv2.putText(img, f"Median Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Markers: {len(image_poses_list)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Mean Error: {np.mean(errors):.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw individual marker poses (without color parameter for compatibility)
                for i, (rvec, tvec) in enumerate(image_poses_list):
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
            
            # Save debug image
            debug_path = f"calibration/out/imgs/debug_aruco_{img_path.name}"
            cv2.imwrite(debug_path, img)
            print(f"  Saved debug image with pose visualization")
        else:
            print(f"  No markers detected")
    
    print(f"Total poses estimated: {len(poses)}")
    print(f"Detected marker IDs: {marker_ids}")
    
    return poses, marker_ids, image_poses, pose_errors

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
        'image_poses': image_poses,
        'pose_errors': pose_errors,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration results saved to {output_file}")

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
            args.aruco_dir, camera_matrix, dist_coeffs, args.marker_size
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
            mean_errors = [data['mean_error'] for data in pose_errors.values()]
            max_errors = [data['max_error'] for data in pose_errors.values()]
            print(f"  Average mean error: {np.mean(mean_errors):.2f}°")
            print(f"  Average max error: {np.mean(max_errors):.2f}°")
            print(f"  Best image: {min(pose_errors.items(), key=lambda x: x[1]['mean_error'])[0]}")
            print(f"  Worst image: {max(pose_errors.items(), key=lambda x: x[1]['mean_error'])[0]}")
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 