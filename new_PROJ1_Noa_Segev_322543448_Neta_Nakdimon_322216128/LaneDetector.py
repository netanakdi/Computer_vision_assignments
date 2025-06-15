import cv2
import numpy as np

class LaneDetector:
    def __init__(self, video_path):
        """
        Initialize the LaneDetector with a video file path
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
    
    def preprocess_frame(self, frame):
        """
        Preprocess the frame to focus on the road portion
        """
        height = frame.shape[0]
        width = frame.shape[1]
        
        # Crop the top portion of the frame to focus on the road
        cropped_frame = frame[int(height * 0.5):height, 0:width]
        
        # Convert to LAB color space and normalize brightness
        lab = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        normalized_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized_frame

    def process_frame(self, frame):
        """
        Process a single frame to detect lane lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny Edge Detection
        mean_brightness = np.mean(gray)
        low_threshold = int(max(50, mean_brightness * 0.66))
        high_threshold = int(max(150, mean_brightness * 1.33))
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        
        # Define region of interest
        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        
        # Define polygon for the region of interest
        polygon = np.array([[
            (0, height),                    # Bottom left
            (0, int(height * 0.65)),        # Top left
            (width, int(height * 0.65)),    # Top right
            (width, height)                 # Bottom right
        ]], np.int32)
        
        # Apply the mask
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=2,              # Distance resolution in pixels
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=50,       # Minimum number of intersections
            minLineLength=30,   # Minimum length of line
            maxLineGap=100      # Maximum allowed gap between line segments
        )
        
        return lines
    
    def visualize_lanes(self, frame, lines):
        """
        Visualize detected lane lines and fill the lane area
        """
        # Create separate images for lines and fill
        line_image = np.zeros_like(frame)
        fill_image = np.zeros_like(frame)
    
        if lines is not None:
         # Separate lines into left and right lanes
            left_lines = []
            right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Filter out horizontal lines
                if abs(slope) > 0.5:
                    # Categorize lines based on slope and position
                    if slope < 0:  # Left lane
                        left_lines.append(line[0])
                    else:  # Right lane
                        right_lines.append(line[0])

        # Function to average lines
        def average_lines(lines):
            if len(lines) > 0:
                avg_line = np.mean(lines, axis=0)
                x1, y1, x2, y2 = map(int, avg_line)
                
                # Extend the line to the bottom of the frame
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    b = y1 - slope * x1
                    
                    # Bottom point
                    y_bottom = frame.shape[0]
                    x_bottom = int((y_bottom - b) / slope)
                    
                    # Top point
                    y_top = int(frame.shape[0] * 0.6)  # Adjust this value to change height
                    x_top = int((y_top - b) / slope)
                    
                    return [(x_bottom, y_bottom), (x_top, y_top)]
            return None

        # Get averaged lines
        left_lane = average_lines(left_lines)
        right_lane = average_lines(right_lines)

        # Stabilize lanes using a buffer
        buffer_size = 5
        if hasattr(self, 'left_lane_buffer') and hasattr(self, 'right_lane_buffer'):
            if left_lane:
                self.left_lane_buffer.append(left_lane)
                if len(self.left_lane_buffer) > buffer_size:
                    self.left_lane_buffer.pop(0)
                left_lane = np.mean(self.left_lane_buffer, axis=0).astype(int).tolist()

            if right_lane:
                self.right_lane_buffer.append(right_lane)
                if len(self.right_lane_buffer) > buffer_size:
                    self.right_lane_buffer.pop(0)
                right_lane = np.mean(self.right_lane_buffer, axis=0).astype(int).tolist()
        else:
            # Initialize buffers if not present
            self.left_lane_buffer = [left_lane] if left_lane else []
            self.right_lane_buffer = [right_lane] if right_lane else []

        # Draw the lane area if both lanes are detected
        if left_lane and right_lane:
            # Create polygon points
            points = np.array([
                left_lane[0],    # Bottom left
                left_lane[1],    # Top left
                right_lane[1],   # Top right
                right_lane[0]    # Bottom right
            ], np.int32)
            
            # Fill the lane area
            cv2.fillPoly(fill_image, [points], (255, 0, 0))  # Red fill
            
            # Draw the lane lines
            cv2.line(line_image, left_lane[0], left_lane[1], (0, 0, 255), 5)   # Red line
            cv2.line(line_image, right_lane[0], right_lane[1], (0, 0, 255), 5)  # Red line

        # Combine the images
        # First add the semi-transparent fill
        result = cv2.addWeighted(frame, 1, fill_image, 0.3, 0)
        # Then add the solid lines
        result = cv2.addWeighted(result, 1, line_image, 1, 0)
    
        return result


    def process_video(self, output_path=None):
        """
        Process the entire video and optionally save the output
        """
        # Set up video writer if output path is provided
        if output_path:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Preprocess the frame
            processed_frame = self.preprocess_frame(frame)
            
            # Detect lane lines
            lines = self.process_frame(processed_frame)
            
            # Visualize the detected lanes
            frame_with_lanes = self.visualize_lanes(processed_frame, lines)
            
            # Display the result
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            # Save the frame if output path is provided
            if output_path:
                # Resize frame_with_lanes to match original dimensions if needed
                frame_with_lanes = cv2.resize(frame_with_lanes, (frame_width, frame_height))
                out.write(frame_with_lanes)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.release_video()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def release_video(self):
        """
        Release the video capture object
        """
        self.cap.release()
