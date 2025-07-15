#!/usr/bin/env python3
"""
3D visualization of camera calibration results for Prusa MK4.
Shows camera position, frustum, real marker locations, and printer bed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_calibration_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def rotation_vector_to_matrix(rvec):
    rvec = np.array(rvec).reshape(3, 1)
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat

def get_camera_position_and_orientation(rvec, tvec):
    rmat = rotation_vector_to_matrix(rvec)
    tvec = np.array(tvec).reshape(3, 1)
    camera_pos = -rmat.T @ tvec
    camera_rmat = rmat.T
    return camera_pos.flatten(), camera_rmat

def get_real_marker_positions():
    # Marker positions in meters (convert mm to m)
    # ID: (X, Y)
    marker_mm = [
        (40, 40),   # ID 0
        (40, 105),  # ID 1
        (40, 170),  # ID 2
        (220, 40),  # ID 3
        (220, 105), # ID 4
        (220, 170), # ID 5
    ]
    marker_xyz = [(x/1000.0, y/1000.0, 0.0) for x, y in marker_mm]
    return np.array(marker_xyz)

def draw_printer_bed(ax, size_x=0.25, size_y=0.21, color='cyan', alpha=0.15):
    # Draw a rectangle on the XY plane at Z=0
    corners = np.array([
        [0, 0, 0],
        [size_x, 0, 0],
        [size_x, size_y, 0],
        [0, size_y, 0],
    ])
    verts = [corners]
    bed = Poly3DCollection(verts, facecolors=color, alpha=alpha, zorder=0)
    ax.add_collection3d(bed)
    # Draw edges
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1)%4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color=color, alpha=0.5, linewidth=2)

def draw_camera_frustum_simple(ax, cam_pos, cam_rmat, scale=0.1, color='purple'):
    forward = cam_rmat[:, 2]
    right = cam_rmat[:, 0]
    up = cam_rmat[:, 1]
    tip = cam_pos
    base_center = cam_pos + forward * scale
    corners = [
        base_center + right * scale * 0.3 + up * scale * 0.2,
        base_center - right * scale * 0.3 + up * scale * 0.2,
        base_center - right * scale * 0.3 - up * scale * 0.2,
        base_center + right * scale * 0.3 - up * scale * 0.2
    ]
    for corner in corners:
        ax.plot([tip[0], corner[0]], [tip[1], corner[1]], [tip[2], corner[2]], 
               color=color, alpha=0.7, linewidth=2)
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1)%4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
               color=color, alpha=0.7, linewidth=2)

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def visualize_calibration(json_file, image_name):
    data = load_calibration_data(json_file)
    if image_name not in data['image_poses']:
        print(f"Image {image_name} not found in calibration data")
        return
    image_data = data['image_poses'][image_name]
    rvec = np.array(image_data['median_rvec'])
    tvec = np.array(image_data['median_tvec'])
    num_markers = image_data['num_markers']
    camera_pos, camera_rmat = get_camera_position_and_orientation(rvec, tvec)
    marker_positions = get_real_marker_positions()
    # 3D plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Printer bed
    draw_printer_bed(ax, size_x=0.25, size_y=0.21, color='cyan', alpha=0.15)
    # Markers
    ax.scatter(marker_positions[:, 0], marker_positions[:, 1], marker_positions[:, 2],
               c='orange', s=120, marker='s', label='ArUco Markers', edgecolors='black', linewidth=2)
    for i, pos in enumerate(marker_positions):
        ax.text(pos[0], pos[1], pos[2], f'ID:{i}',
                fontsize=12, ha='center', va='bottom', weight='bold')
        # Draw 40mm axis lines from each marker center
        axis_len = 0.04  # 40mm
        # X+ (red)
        ax.plot([pos[0], pos[0]+axis_len], [pos[1], pos[1]], [pos[2], pos[2]], color='r', linewidth=2)
        # Y+ (green)
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]+axis_len], [pos[2], pos[2]], color='g', linewidth=2)
        # Z+ (blue)
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2]+axis_len], color='b', linewidth=2)
        # Draw 40x40mm square (marker outline) in XY plane
        half_size = 0.02  # 20mm
        square_xy = np.array([
            [pos[0]-half_size, pos[1]-half_size, pos[2]],
            [pos[0]+half_size, pos[1]-half_size, pos[2]],
            [pos[0]+half_size, pos[1]+half_size, pos[2]],
            [pos[0]-half_size, pos[1]+half_size, pos[2]],
            [pos[0]-half_size, pos[1]-half_size, pos[2]],
        ])
        ax.plot(square_xy[:,0], square_xy[:,1], square_xy[:,2], color='k', linewidth=1.5)
    # Camera
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
               c='red', s=200, marker='o', label='Camera Position', edgecolors='black', linewidth=2)
    draw_camera_frustum_simple(ax, camera_pos, camera_rmat, scale=0.15, color='purple')
    # Viewing rays
    for i, pos in enumerate(marker_positions):
        ax.plot([camera_pos[0], pos[0]], [camera_pos[1], pos[1]], [camera_pos[2], pos[2]],
               'b--', alpha=0.6, linewidth=1)
    # Set plot properties
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(f'Camera Calibration Visualization\nImage: {image_name}', fontsize=14, weight='bold')
    ax.set_box_aspect([1, 1, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=25, azim=45)
    set_axes_equal(ax)
    ax.text2D(0.02, 0.95, f'Camera Position: ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    pose_errors = data['pose_errors'].get(image_name, {})
    if pose_errors:
        mean_error = pose_errors.get('mean_error', 0)
        max_error = pose_errors.get('max_error', 0)
        ax.text2D(0.02, 0.88, f'Mean Error: {mean_error:.2f}°\nMax Error: {max_error:.2f}°',
                 transform=ax.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    ax.text2D(0.02, 0.78, f'Detected Markers: {num_markers}',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.show()

def main():
    json_file = "../out/camera_calibration.json"
    image_name = "PXL_20250710_145216379.jpg"
    if not Path(json_file).exists():
        print(f"Calibration file not found: {json_file}")
        return
    print(f"Visualizing calibration for image: {image_name}")
    print(f"Loading data from: {json_file}")
    try:
        visualize_calibration(json_file, image_name)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 