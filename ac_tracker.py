import pandas as pd
import numpy as np
import pygame
import sys
import time
import os
from collections import defaultdict

class CarVisualizer:
    def __init__(self, csv_path, width=1920, height=1080):
        self.csv_path = csv_path
        self.width = width
        self.height = height
        self.last_modified_time = 0
        self.car_colors = {}  # Dictionary to store consistent colors for each car
        self.dot_radius = 10
        self.last_frame_data = None  # Store previous frame data
        
        # Initialize pygame with transparency support
        pygame.init()
        
        # Try to create a transparent window (platform-dependent)
        try:
            # For Windows, this might work
            import platform
            if platform.system() == "Windows":
                import win32api
                import win32con
                import win32gui
                
                # Create pygame window
                self.screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
                
                # Get window handle
                hwnd = pygame.display.get_wm_info()["window"]
                
                # Set window to be transparent and always on top
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                      win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | 
                                      win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST)
                
                # Set transparency color (we'll use pure black)
                win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
                
                self.transparent_mode = True
                self.transparent_color = (0, 0, 0)  # Pure black will be transparent
                
        except:
            # Fallback for other platforms or if win32 not available
            self.screen = pygame.display.set_mode((width, height))
            self.transparent_mode = False
            self.transparent_color = (30, 30, 30)  # Dark gray background
            print("Note: Transparent mode not available. Install pywin32 for transparency on Windows.")
        
        pygame.display.set_caption("AC Car Position Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Generate distinct colors for cars (avoiding pure black for transparency)
        self.color_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 255, 255),# White
            (128, 255, 0),  # Lime
        ]
    
    def project_to_screen(self, car_pos, camera_pos, camera_look, fov):
        """Project 3D car position to 2D screen coordinates"""
        # Calculate vector from camera to car
        to_car = car_pos - camera_pos
        
        # Normalize camera look direction
        look_dir = camera_look / np.linalg.norm(camera_look)
        
        # Calculate camera up vector (assuming mostly upright camera)
        world_up = np.array([0, 1, 0])
        right = np.cross(look_dir, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, look_dir)
        
        # Create camera rotation matrix
        rotation_matrix = np.array([right, up, -look_dir])
        
        # Transform to camera space
        camera_space = rotation_matrix @ to_car
        
        # If behind camera, don't render
        if camera_space[2] >= 0:
            return None
        
        # Convert FOV to radians
        fov_rad = np.radians(fov)
        
        # Calculate projection
        aspect_ratio = self.width / self.height
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        # Project to normalized device coordinates
        x_ndc = (f / aspect_ratio) * camera_space[0] / -camera_space[2]
        y_ndc = f * camera_space[1] / -camera_space[2]
        
        # Convert to screen coordinates
        screen_x = (x_ndc + 1) * 0.5 * self.width
        screen_y = (1 - y_ndc) * 0.5 * self.height  # Flip Y axis
        
        return int(screen_x), int(screen_y)
    
    def get_latest_frame(self):
        """Read the CSV and return data for the latest timestamp"""
        try:
            # Check if file has been modified
            current_modified_time = os.path.getmtime(self.csv_path)
            if current_modified_time == self.last_modified_time:
                return self.last_frame_data  # Return cached data if file hasn't changed
            
            self.last_modified_time = current_modified_time
            
            # Read CSV
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return self.last_frame_data
            
            # Get the latest timestamp
            latest_time = df['timestamp'].max()
            
            # Filter for the latest timestamp
            latest_frame = df[df['timestamp'] == latest_time]
            
            # Update cached data
            self.last_frame_data = latest_frame
            
            return latest_frame
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return self.last_frame_data  # Return last known good data
    
    def assign_car_color(self, car_id):
        """Assign a consistent color to each car"""
        if car_id not in self.car_colors:
            color_index = len(self.car_colors) % len(self.color_palette)
            self.car_colors[car_id] = self.color_palette[color_index]
        return self.car_colors[car_id]
    
    def run(self):
        """Main visualization loop"""
        running = True
        last_update = time.time()
        fps_update_interval = 0.5
        fps = 0
        frame_count = 0
        
        print("Visualizer running...")
        if self.transparent_mode:
            print("Running in transparent mode - overlay on top of AC")
        else:
            print("Running in normal mode")
        print("Press ESC or close window to exit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen with transparent color
            self.screen.fill(self.transparent_color)
            
            # Get latest frame data (will return cached data if file hasn't changed)
            frame_data = self.get_latest_frame()
            
            if frame_data is not None and not frame_data.empty:
                # Extract camera data from first row (same for all cars in a frame)
                camera_pos = np.array([
                    frame_data.iloc[0]['camera_pos_x'],
                    frame_data.iloc[0]['camera_pos_y'],
                    frame_data.iloc[0]['camera_pos_z']
                ])
                camera_look = np.array([
                    frame_data.iloc[0]['camera_look_x'],
                    frame_data.iloc[0]['camera_look_y'],
                    frame_data.iloc[0]['camera_look_z']
                ])
                fov = frame_data.iloc[0]['camera_fov']
                
                # Project and draw each car
                for _, row in frame_data.iterrows():
                    car_id = int(row['car_id'])
                    car_pos = np.array([row['x'], row['y'], row['z']])
                    
                    # Project to screen
                    screen_coords = self.project_to_screen(car_pos, camera_pos, camera_look, fov)
                    
                    if screen_coords is not None:
                        x, y = screen_coords
                        # Only draw if within screen bounds (with some margin)
                        if -50 <= x <= self.width + 50 and -50 <= y <= self.height + 50:
                            color = self.assign_car_color(car_id)
                            pygame.draw.circle(self.screen, color, (x, y), self.dot_radius)
                            
                            # Draw car ID next to dot
                            label = self.font.render(str(car_id), True, color)
                            self.screen.blit(label, (x + 15, y - 10))
                
                # Display timestamp (avoid pure black for text in transparent mode)
                timestamp = frame_data.iloc[0]['timestamp']
                text_color = (255, 255, 255) if not self.transparent_mode or self.transparent_color != (255, 255, 255) else (200, 200, 200)
                time_text = self.font.render(f"Time: {timestamp:.2f}s", True, text_color)
                self.screen.blit(time_text, (10, 10))
            
            # Calculate and display FPS
            frame_count += 1
            if time.time() - last_update > fps_update_interval:
                fps = frame_count / (time.time() - last_update)
                frame_count = 0
                last_update = time.time()
            
            text_color = (255, 255, 255) if not self.transparent_mode or self.transparent_color != (255, 255, 255) else (200, 200, 200)
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, text_color)
            self.screen.blit(fps_text, (10, 50))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Cap at 60 FPS
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Path to your CSV file
    csv_path = os.path.expanduser("~/Documents/Assetto Corsa/car_position.csv")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        print("Please make sure the path is correct and the file exists.")
        sys.exit(1)
    
    # Create and run visualizer
    visualizer = CarVisualizer(csv_path)
    visualizer.run()