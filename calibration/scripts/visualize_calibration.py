#!/usr/bin/env python3
"""
Interactive 3D visualization of camera calibration results for Prusa MK4.
Shows camera position, frustum, real marker locations, printer bed, and source images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button

class CalibrationVisualizer:
    def __init__(self, json_file, images_dir):
        self.json_file = json_file
        self.images_dir = Path(images_dir)
        self.data = self.load_calibration_data()
        self.image_names = list(self.data['image_poses'].keys())
        self.current_image_idx = 0
        self.fig = None
        self.ax_3d = None
        self.ax_img = None
        
    def load_calibration_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        return data
    
    def rotation_vector_to_matrix(self, rvec):
        rvec = np.array(rvec).reshape(3, 1)
        rmat, _ = cv2.Rodrigues(rvec)
        return rmat

    def get_camera_position_and_orientation(self, rvec, tvec, flip_y=True, flip_z=True):
        rmat = self.rotation_vector_to_matrix(rvec)
        tvec = np.array(tvec).reshape(3, 1)
        
        # Apply coordinate system transformation
        transform = np.eye(3)
        if flip_y:
            transform[1, 1] = -1  # Flip Y axis
        if flip_z:
            transform[2, 2] = -1  # Flip Z axis
        
        # Transform rotation and translation
        rmat_transformed = transform @ rmat @ transform.T
        tvec_transformed = transform @ tvec
        
        # Camera position in world coordinates
        camera_pos = -rmat_transformed.T @ tvec_transformed
        camera_rmat = rmat_transformed.T
        
        return camera_pos.flatten(), camera_rmat

    def get_real_marker_positions(self):
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

    def draw_printer_bed(self, ax, size_x=0.25, size_y=0.21, color='cyan', alpha=0.15):
        corners = np.array([
            [0, 0, 0],
            [size_x, 0, 0],
            [size_x, size_y, 0],
            [0, size_y, 0],
        ])
        verts = [corners]
        bed = Poly3DCollection(verts, facecolors=color, alpha=alpha, zorder=0)
        ax.add_collection3d(bed)
        for i in range(4):
            p1, p2 = corners[i], corners[(i+1)%4]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color=color, alpha=0.5, linewidth=2)

    def draw_camera_frustum_simple(self, ax, cam_pos, cam_rmat, scale=0.1, color='purple'):
        forward = -cam_rmat[:, 2]  # Negative Z is camera forward direction
        right = cam_rmat[:, 0]     # X is right
        up = -cam_rmat[:, 1]       # Negative Y is up (camera Y is typically down)
        
        tip = cam_pos
        base_center = cam_pos + forward * scale
        corners = [
            base_center + right * scale * 0.5 + up * scale * 0.4,
            base_center - right * scale * 0.5 + up * scale * 0.4,
            base_center - right * scale * 0.5 - up * scale * 0.4,
            base_center + right * scale * 0.5 - up * scale * 0.4
        ]
        for corner in corners:
            ax.plot([tip[0], corner[0]], [tip[1], corner[1]], [tip[2], corner[2]], 
                   color=color, alpha=0.8, linewidth=3)
        for i in range(4):
            p1, p2 = corners[i], corners[(i+1)%4]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   color=color, alpha=0.8, linewidth=2)

    def calculate_marker_centroid(self, marker_positions):
        return np.mean(marker_positions, axis=0)

    def set_axes_equal(self, ax):
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

    def load_source_image(self, image_name):
        """Load the source calibration image"""
        # Try both the original images and debug images
        possible_paths = [
            self.images_dir.parent / "aruco" / image_name,
            self.images_dir / f"debug_aruco_{image_name}",
            self.images_dir / image_name
        ]
        
        for path in possible_paths:
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    # Convert BGR to RGB for matplotlib
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def update_visualization(self):
        """Update the 3D visualization and source image for current image"""
        image_name = self.image_names[self.current_image_idx]
        
        # Clear previous plots
        self.ax_3d.clear()
        self.ax_img.clear()
        
        # Get calibration data for current image
        image_data = self.data['image_poses'][image_name]
        rvec = np.array(image_data['median_rvec'])
        tvec = np.array(image_data['median_tvec'])
        num_markers = image_data['num_markers']
        camera_pos, camera_rmat = self.get_camera_position_and_orientation(rvec, tvec)
        marker_positions = self.get_real_marker_positions()
        
        # 3D Plot
        self.draw_printer_bed(self.ax_3d, size_x=0.25, size_y=0.21, color='cyan', alpha=0.15)
        
        # Markers
        self.ax_3d.scatter(marker_positions[:, 0], marker_positions[:, 1], marker_positions[:, 2],
                   c='orange', s=120, marker='s', label='ArUco Markers', edgecolors='black', linewidth=2)
        for i, pos in enumerate(marker_positions):
            self.ax_3d.text(pos[0], pos[1], pos[2], f'ID:{i}',
                    fontsize=12, ha='center', va='bottom', weight='bold')
            # Draw 40mm axis lines from each marker center
            axis_len = 0.04  # 40mm
            # X+ (red), Y+ (green), Z+ (blue)
            self.ax_3d.plot([pos[0], pos[0]+axis_len], [pos[1], pos[1]], [pos[2], pos[2]], color='r', linewidth=2)
            self.ax_3d.plot([pos[0], pos[0]], [pos[1], pos[1]+axis_len], [pos[2], pos[2]], color='g', linewidth=2)
            self.ax_3d.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2]+axis_len], color='b', linewidth=2)
            # Draw 40x40mm square (marker outline) in XY plane
            half_size = 0.02  # 20mm
            square_xy = np.array([
                [pos[0]-half_size, pos[1]-half_size, pos[2]],
                [pos[0]+half_size, pos[1]-half_size, pos[2]],
                [pos[0]+half_size, pos[1]+half_size, pos[2]],
                [pos[0]-half_size, pos[1]+half_size, pos[2]],
                [pos[0]-half_size, pos[1]-half_size, pos[2]],
            ])
            self.ax_3d.plot(square_xy[:,0], square_xy[:,1], square_xy[:,2], color='k', linewidth=1.5)
            
        # Camera
        self.ax_3d.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
                   c='red', s=200, marker='o', label='Camera Position', edgecolors='black', linewidth=2)
        self.draw_camera_frustum_simple(self.ax_3d, camera_pos, camera_rmat, scale=0.25, color='purple')
        
        # Draw line from camera to marker centroid
        marker_centroid = self.calculate_marker_centroid(marker_positions)
        self.ax_3d.plot([camera_pos[0], marker_centroid[0]], [camera_pos[1], marker_centroid[1]], [camera_pos[2], marker_centroid[2]],
               'yellow', linewidth=3, alpha=0.8, label='Camera to Markers')
        
        # Viewing rays
        for i, pos in enumerate(marker_positions):
            self.ax_3d.plot([camera_pos[0], pos[0]], [camera_pos[1], pos[1]], [camera_pos[2], pos[2]],
                   'b--', alpha=0.6, linewidth=1)
        
        # Set 3D plot properties
        self.ax_3d.set_xlabel('X (meters)', fontsize=12)
        self.ax_3d.set_ylabel('Y (meters)', fontsize=12)
        self.ax_3d.set_zlabel('Z (meters)', fontsize=12)
        self.ax_3d.set_title(f'Camera Calibration Visualization\nImage: {image_name}', fontsize=14, weight='bold')
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_3d.legend(fontsize=10)
        self.ax_3d.grid(True, alpha=0.3)
        self.ax_3d.view_init(elev=25, azim=45)
        self.set_axes_equal(self.ax_3d)
        
        # Load and display source image
        source_img = self.load_source_image(image_name)
        if source_img is not None:
            self.ax_img.imshow(source_img)
            self.ax_img.set_title(f'Source Image: {image_name}', fontsize=12, weight='bold')
        else:
            self.ax_img.text(0.5, 0.5, f'Image not found:\n{image_name}', 
                    ha='center', va='center', transform=self.ax_img.transAxes, fontsize=12)
            self.ax_img.set_title(f'Image not found: {image_name}', fontsize=12)
        
        self.ax_img.axis('off')
        
        # Add statistics
        pose_errors = self.data['pose_errors'].get(image_name, {})
        if pose_errors:
            mean_error = pose_errors.get('mean_error', 0)
            max_error = pose_errors.get('max_error', 0)
            info_text = f'Mean Error: {mean_error:.2f}°\nMax Error: {max_error:.2f}°\nMarkers: {num_markers}'
        else:
            info_text = f'Markers: {num_markers}'
            
        self.fig.text(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Update image counter
        self.fig.text(0.5, 0.95, f'Image {self.current_image_idx + 1} of {len(self.image_names)}', 
                ha='center', fontsize=12, weight='bold')
        
        self.fig.canvas.draw()

    def next_image(self, event):
        """Go to next image"""
        self.current_image_idx = (self.current_image_idx + 1) % len(self.image_names)
        self.update_visualization()
    
    def prev_image(self, event):
        """Go to previous image"""
        self.current_image_idx = (self.current_image_idx - 1) % len(self.image_names)
        self.update_visualization()

    def visualize(self):
        """Create the interactive visualization"""
        if not self.image_names:
            print("No images found in calibration data")
            return
            
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        
        # 3D plot on the left (larger)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        
        # Image on the right
        self.ax_img = self.fig.add_subplot(122)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.4, 0.01, 0.08, 0.04])
        ax_next = plt.axes([0.52, 0.01, 0.08, 0.04])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        
        btn_prev.on_clicked(self.prev_image)
        btn_next.on_clicked(self.next_image)
        
        # Initial visualization
        self.update_visualization()
        
        plt.tight_layout()
        plt.show()

def main():
    json_file = "../out/camera_calibration.json"
    images_dir = "../out/imgs"  # Directory containing debug images
    
    if not Path(json_file).exists():
        print(f"Calibration file not found: {json_file}")
        return
    
    print(f"Loading calibration data from: {json_file}")
    print(f"Looking for images in: {images_dir}")
    print("Use Previous/Next buttons to navigate between images")
    
    try:
        visualizer = CalibrationVisualizer(json_file, images_dir)
        visualizer.visualize()
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 