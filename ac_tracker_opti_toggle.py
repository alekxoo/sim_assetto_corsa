import pandas as pd
import numpy as np
import pygame
import sys
import time
import os
from collections import defaultdict
import threading
import queue

class CarVisualizer:
    def __init__(self, csv_path, width=1920, height=1080):
        self.csv_path = csv_path
        self.width = width
        self.height = height
        self.car_colors = {}  # Dictionary to store consistent colors for each car
        self.dot_radius = 10
        self.last_frame_data = None  # Store previous frame data
        self.show_overlay = True  # Toggle for showing dots, labels, and FPS
        
        # File reading optimization
        self.file_position = 0  # Track where we left off in the file
        self.data_queue = queue.Queue(maxsize=10)  # Buffer for processed data
        self.reader_thread = None
        self.running = True
        
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
        self.time_font = pygame.font.Font(None, 48)  # Larger font for time display
        
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
        
        # Start the file reader thread
        self.start_reader_thread()
    
    def start_reader_thread(self):
        """Start a background thread to read new CSV data"""
        self.reader_thread = threading.Thread(target=self.csv_reader_worker, daemon=True)
        self.reader_thread.start()
    
    def csv_reader_worker(self):
        """Background worker that reads only new lines from the CSV"""
        header_line = None
        column_names = None
        
        while self.running:
            try:
                # Check if file exists and has been modified
                if not os.path.exists(self.csv_path):
                    time.sleep(0.1)
                    continue
                
                file_size = os.path.getsize(self.csv_path)
                
                # If file is smaller than our position, it was probably recreated
                if file_size < self.file_position:
                    self.file_position = 0
                    header_line = None
                
                # If no new data, wait
                if file_size == self.file_position:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    continue
                
                # Read only the new part of the file
                with open(self.csv_path, 'r') as f:
                    f.seek(self.file_position)
                    
                    # If we're at the beginning, read the header
                    if self.file_position == 0:
                        header_line = f.readline()
                        column_names = header_line.strip().split(',')
                        self.file_position = f.tell()
                        continue
                    
                    # Read new lines
                    new_lines = []
                    for line in f:
                        if line.strip():  # Skip empty lines
                            new_lines.append(line.strip())
                    
                    # Update file position
                    self.file_position = f.tell()
                    
                    if new_lines and column_names:
                        # Parse the new data
                        data_dict = {col: [] for col in column_names}
                        
                        for line in new_lines:
                            values = line.split(',')
                            if len(values) == len(column_names):
                                for col, val in zip(column_names, values):
                                    try:
                                        # Try to convert to float, otherwise keep as string
                                        data_dict[col].append(float(val))
                                    except ValueError:
                                        data_dict[col].append(val)
                        
                        # Create DataFrame from the new data
                        df = pd.DataFrame(data_dict)
                        
                        if not df.empty:
                            # Get the latest timestamp from this batch
                            latest_time = df['timestamp'].max()
                            latest_frame = df[df['timestamp'] == latest_time]
                            
                            # Put in queue (non-blocking)
                            try:
                                self.data_queue.put_nowait(latest_frame)
                            except queue.Full:
                                # If queue is full, remove oldest and add new
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(latest_frame)
                                except:
                                    pass
                
            except Exception as e:
                print(f"Reader thread error: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self):
        """Get the latest frame data from the queue"""
        latest_data = None
        
        # Empty the queue and keep only the most recent data
        while not self.data_queue.empty():
            try:
                latest_data = self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        # If we got new data, update our cached frame
        if latest_data is not None:
            self.last_frame_data = latest_data
        
        return self.last_frame_data
    
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
        print("Press \\ to toggle overlay elements")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.running = False  # Stop reader thread
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        self.running = False  # Stop reader thread
                    elif event.key == pygame.K_BACKSLASH:  # Toggle overlay with backslash key
                        self.show_overlay = not self.show_overlay
            
            # Clear screen with transparent color
            self.screen.fill(self.transparent_color)
            
            # Get latest frame data from queue
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
                
                # Only draw dots and labels if overlay is enabled
                if self.show_overlay:
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
                
                # Always display timestamp with enhanced visibility
                timestamp = frame_data.iloc[0]['timestamp']
                time_text = self.time_font.render(f"Time: {timestamp:.2f}s", True, (255, 255, 255))
                
                # Get text dimensions for background
                text_rect = time_text.get_rect()
                text_rect.topleft = (10, 10)
                
                # Draw black background rectangle with padding
                padding = 10
                background_rect = pygame.Rect(
                    text_rect.left - padding,
                    text_rect.top - padding,
                    text_rect.width + 2 * padding,
                    text_rect.height + 2 * padding
                )
                
                # Draw background (not pure black if in transparent mode)
                bg_color = (1, 1, 1) if self.transparent_mode else (0, 0, 0)
                pygame.draw.rect(self.screen, bg_color, background_rect)
                
                # Draw the time text
                self.screen.blit(time_text, text_rect)
            
            # Only show FPS if overlay is enabled
            if self.show_overlay:
                # Calculate and display FPS
                frame_count += 1
                if time.time() - last_update > fps_update_interval:
                    fps = frame_count / (time.time() - last_update)
                    frame_count = 0
                    last_update = time.time()
                
                fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
                self.screen.blit(fps_text, (10, 80))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Cap at 60 FPS
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Path to your CSV file
    csv_path = os.path.expanduser("~/Documents/Assetto Corsa/all_car_positions.csv")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        print("Please make sure the path is correct and the file exists.")
        print("Starting anyway - will wait for file to be created...")
    
    # Create and run visualizer
    visualizer = CarVisualizer(csv_path)
    visualizer.run()