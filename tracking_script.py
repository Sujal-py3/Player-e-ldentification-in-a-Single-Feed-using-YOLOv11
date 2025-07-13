
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedPlayerTracker:
    def __init__(self, model_path='best.pt', max_disappeared=12, max_distance=60, min_confidence=0.65):
        """
        Advanced player tracker with improved stability and re-identification
        
        Args:
            model_path: Path to the YOLO model
            max_disappeared: Maximum frames a player can be missing
            max_distance: Maximum distance for player matching
            min_confidence: Minimum detection confidence
        """
        self.model = YOLO(model_path)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        
        # Core tracking variables
        self.next_player_id = 0
        self.active_players = {}
        self.disappeared_players = {}
        
        # Enhanced feature storage
        self.player_features = {}
        self.player_positions = {}
        self.player_velocities = {}
        self.feature_history_size = 6
        self.position_history_size = 4
        
        # Stability and quality control
        self.player_stability = {}
        self.player_quality_scores = {}
        self.min_stability_frames = 2
        self.stability_threshold = 0.5
        
        # Detection filtering
        self.detection_history = []
        self.max_detection_history = 3
        
        # Frame counter
        self.frame_count = 0
        
    def is_valid_player_detection(self, bbox, confidence, frame_shape):
        """
        Enhanced validation for player detections
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            frame_shape: (height, width) of frame
            
        Returns:
            Boolean and quality score
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        quality_score = confidence
        
        # Size validation
        if width < 15 or height < 30:
            return False, 0.0
        if width > 150 or height > 300:
            return False, 0.0
            
        # Aspect ratio validation (players should be taller than wide)
        aspect_ratio = height / width
        if aspect_ratio < 1.0 or aspect_ratio > 5.0:
            return False, 0.0
            
        # Penalize extreme aspect ratios
        if aspect_ratio < 1.5 or aspect_ratio > 3.5:
            quality_score *= 0.8
            
        # Position validation
        frame_height, frame_width = frame_shape[:2]
        
        # Check if detection is in reasonable field area
        if y2 > frame_height * 0.98:  # Too close to bottom
            return False, 0.0
        if y1 < frame_height * 0.05:  # Too close to top (crowd area)
            return False, 0.0
        if x1 < 10 or x2 > frame_width - 10:  # Too close to sides
            quality_score *= 0.9
            
        # Size consistency check
        area = width * height
        if area < 500:  # Too small
            quality_score *= 0.7
        if area > 15000:  # Too large
            quality_score *= 0.8
            
        return quality_score > 0.3, quality_score
    
    def extract_player_features(self, frame, bbox):
        """
        Extract comprehensive features for player identification
        
        Args:
            frame: Current frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Feature vector
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within frame
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(x1, frame_width-1))
        y1 = max(0, min(y1, frame_height-1))
        x2 = max(x1+1, min(x2, frame_width))
        y2 = max(y1+1, min(y2, frame_height))
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(48)
        
        try:
            # Resize to standard size
            player_region = cv2.resize(player_region, (24, 48))
            
            # Convert to different color spaces
            hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
            
            # Focus on torso area for jersey color (upper 60% of detection)
            torso_height = int(player_region.shape[0] * 0.6)
            torso_region = hsv_region[:torso_height, :]
            
            features = []
            
            # 1. Color histogram features (jersey identification)
            if torso_region.size > 0:
                # HSV histograms for jersey colors
                hist_h = cv2.calcHist([torso_region], [0], None, [6], [0, 180])
                hist_s = cv2.calcHist([torso_region], [1], None, [6], [0, 256])
                hist_v = cv2.calcHist([torso_region], [2], None, [6], [0, 256])
                
                # Normalize histograms
                hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
                hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
                hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
                
                features.extend(hist_h)  # 6 features
                features.extend(hist_s)  # 6 features
                features.extend(hist_v[:4])  # 4 features
            else:
                features.extend([0] * 16)
            
            # 2. Dominant color extraction
            try:
                pixels = torso_region.reshape(-1, 3)
                if len(pixels) > 10:
                    # Use KMeans to find dominant colors
                    kmeans = KMeans(n_clusters=min(2, len(pixels)), random_state=42, n_init=5)
                    kmeans.fit(pixels)
                    centers = kmeans.cluster_centers_
                    
                    # Add dominant colors as features
                    if len(centers) >= 1:
                        features.extend(centers[0] / 255.0)  # 3 features
                    else:
                        features.extend([0, 0, 0])
                        
                    if len(centers) >= 2:
                        features.extend(centers[1] / 255.0)  # 3 features  
                    else:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0] * 6)
            except:
                features.extend([0] * 6)
            
            # 3. Spatial features
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height
            rel_width = (x2 - x1) / frame_width
            rel_height = (y2 - y1) / frame_height
            
            features.extend([center_x, center_y, rel_width, rel_height])  # 4 features
            
            # 4. Texture features
            gray_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture measures
            if gray_region.size > 0:
                # Standard deviation (texture roughness)
                texture_std = np.std(gray_region) / 255.0
                
                # Edge density
                edges = cv2.Canny(gray_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                features.extend([texture_std, edge_density])  # 2 features
            else:
                features.extend([0, 0])
                
        except Exception as e:
            return np.zeros(48)
        
        # Ensure exactly 48 features
        features = np.array(features)
        if len(features) < 48:
            features = np.pad(features, (0, 48 - len(features)), 'constant')
        elif len(features) > 48:
            features = features[:48]
            
        return features
    
    def calculate_feature_similarity(self, features1, features2, pos1, pos2):
        """
        Calculate similarity between two feature vectors
        
        Args:
            features1, features2: Feature vectors
            pos1, pos2: Position tuples (x, y)
            
        Returns:
            Similarity score (0-1)
        """
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
            
        # 1. Color similarity (most important for jersey matching)
        color_features1 = features1[:22]  # HSV histograms + dominant colors
        color_features2 = features2[:22]
        
        # Cosine similarity for color features
        norm1 = np.linalg.norm(color_features1)
        norm2 = np.linalg.norm(color_features2)
        
        if norm1 > 0 and norm2 > 0:
            color_similarity = np.dot(color_features1, color_features2) / (norm1 * norm2)
        else:
            color_similarity = 0.0
            
        # 2. Position similarity
        pos_distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        pos_similarity = np.exp(-pos_distance / 40)  # Exponential decay
        
        # 3. Size similarity  
        size_features1 = features1[22:26]
        size_features2 = features2[22:26]
        size_distance = np.linalg.norm(size_features1 - size_features2)
        size_similarity = np.exp(-size_distance * 3)
        
        # 4. Texture similarity
        texture_features1 = features1[26:28]
        texture_features2 = features2[26:28]
        texture_distance = np.linalg.norm(texture_features1 - texture_features2)
        texture_similarity = np.exp(-texture_distance * 2)
        
        # Weighted combination
        total_similarity = (0.5 * color_similarity + 
                          0.3 * pos_similarity + 
                          0.15 * size_similarity + 
                          0.05 * texture_similarity)
        
        return max(0, min(1, total_similarity))
    
    def update_player_stability(self, player_id, quality_score):
        """
        Update stability tracking for a player
        
        Args:
            player_id: Player ID
            quality_score: Quality score of current detection
        """
        if player_id not in self.player_stability:
            self.player_stability[player_id] = []
            
        self.player_stability[player_id].append(quality_score)
        
        # Keep only recent stability scores
        if len(self.player_stability[player_id]) > 8:
            self.player_stability[player_id].pop(0)
    
    def get_player_stability(self, player_id):
        """
        Get stability score for a player
        
        Args:
            player_id: Player ID
            
        Returns:
            Stability score (0-1)
        """
        if player_id not in self.player_stability:
            return 0.0
            
        stability_scores = self.player_stability[player_id]
        
        if len(stability_scores) < self.min_stability_frames:
            return 0.0
            
        # Recent stability is more important
        recent_scores = stability_scores[-4:]
        stability = np.mean(recent_scores)
        
        # Bonus for consistency
        if len(stability_scores) >= 4:
            consistency = 1.0 - np.std(stability_scores[-4:])
            stability = stability * 0.8 + consistency * 0.2
            
        return min(1.0, stability)
    
    def register_new_player(self, centroid, bbox, features, confidence, quality_score):
        """
        Register a new player
        
        Args:
            centroid: Center point (x, y)
            bbox: Bounding box
            features: Feature vector
            confidence: Detection confidence
            quality_score: Quality score
            
        Returns:
            player_id
        """
        player_id = self.next_player_id
        
        self.active_players[player_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0,
            'confidence': confidence,
            'first_seen': self.frame_count,
            'last_seen': self.frame_count
        }
        
        self.player_features[player_id] = [features]
        self.player_positions[player_id] = [centroid]
        self.player_velocities[player_id] = [(0, 0)]
        
        self.update_player_stability(player_id, quality_score)
        
        self.next_player_id += 1
        return player_id
    
    def update_player_info(self, player_id, centroid, bbox, features, confidence, quality_score):
        """
        Update existing player information
        
        Args:
            player_id: Player ID
            centroid: New center point
            bbox: New bounding box
            features: New feature vector
            confidence: Detection confidence
            quality_score: Quality score
        """
        # Calculate velocity
        old_centroid = self.active_players[player_id]['centroid']
        velocity = (centroid[0] - old_centroid[0], centroid[1] - old_centroid[1])
        
        # Update player info
        self.active_players[player_id].update({
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0,
            'confidence': confidence,
            'last_seen': self.frame_count
        })
        
        # Update feature history
        self.player_features[player_id].append(features)
        if len(self.player_features[player_id]) > self.feature_history_size:
            self.player_features[player_id].pop(0)
        
        # Update position history
        self.player_positions[player_id].append(centroid)
        if len(self.player_positions[player_id]) > self.position_history_size:
            self.player_positions[player_id].pop(0)
            
        # Update velocity history
        self.player_velocities[player_id].append(velocity)
        if len(self.player_velocities[player_id]) > self.position_history_size:
            self.player_velocities[player_id].pop(0)
        
        # Update stability
        self.update_player_stability(player_id, quality_score)
    
    def match_with_disappeared_players(self, detection):
        """
        Try to match detection with recently disappeared players
        
        Args:
            detection: Detection dictionary
            
        Returns:
            player_id if match found, None otherwise
        """
        if len(self.disappeared_players) == 0:
            return None
            
        best_match_id = None
        best_similarity = 0.4  # Threshold for re-identification
        
        for player_id, player_info in self.disappeared_players.items():
            if player_info['disappeared'] > self.max_disappeared:
                continue
                
            if player_id in self.player_features and len(self.player_features[player_id]) > 0:
                # Use average of stored features
                avg_features = np.mean(self.player_features[player_id], axis=0)
                
                similarity = self.calculate_feature_similarity(
                    avg_features,
                    detection['features'],
                    player_info['centroid'],
                    detection['centroid']
                )
                
                # Consider time since disappeared
                time_penalty = min(0.2, player_info['disappeared'] * 0.02)
                similarity -= time_penalty
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = player_id
        
        return best_match_id
    
    def should_display_player(self, player_id):
        """
        Determine if player should be displayed
        
        Args:
            player_id: Player ID
            
        Returns:
            Boolean
        """
        stability = self.get_player_stability(player_id)
        frames_tracked = self.frame_count - self.active_players[player_id]['first_seen']
        
        # Player must be stable enough and tracked for minimum frames
        return stability >= self.stability_threshold and frames_tracked >= self.min_stability_frames
    
    def update(self, frame):
        """
        Main update function
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary of stable tracked players
        """
        self.frame_count += 1
        
        # Run detection
        results = self.model(frame, verbose=False)
        
        # Extract valid detections
        valid_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                # Validate detection
                is_valid, quality_score = self.is_valid_player_detection(box, conf, frame.shape)
                
                if is_valid and conf >= self.min_confidence:
                    centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    features = self.extract_player_features(frame, box)
                    
                    valid_detections.append({
                        'bbox': box,
                        'centroid': centroid,
                        'confidence': conf,
                        'features': features,
                        'quality_score': quality_score
                    })
        
        # Handle case with no detections
        if len(valid_detections) == 0:
            # Mark all players as disappeared
            for player_id in list(self.active_players.keys()):
                self.active_players[player_id]['disappeared'] += 1
                
                if self.active_players[player_id]['disappeared'] > self.max_disappeared:
                    self.disappeared_players[player_id] = self.active_players[player_id]
                    del self.active_players[player_id]
                    
            return self.get_stable_players()
        
        # Handle case with no existing players
        if len(self.active_players) == 0:
            for detection in valid_detections:
                if detection['confidence'] > 0.7:  # Higher threshold for new players
                    self.register_new_player(
                        detection['centroid'],
                        detection['bbox'],
                        detection['features'],
                        detection['confidence'],
                        detection['quality_score']
                    )
            return self.get_stable_players()
        
        # Match detections to existing players
        self.match_detections_to_players(valid_detections)
        
        # Clean up old disappeared players
        self.cleanup_disappeared_players()
        
        return self.get_stable_players()
    
    def match_detections_to_players(self, detections):
        """
        Match detections to existing players using Hungarian algorithm
        
        Args:
            detections: List of detection dictionaries
        """
        active_ids = list(self.active_players.keys())
        n_players = len(active_ids)
        n_detections = len(detections)
        
        # Create cost matrix
        cost_matrix = np.ones((n_players, n_detections)) * 1000  # High cost for no match
        
        for i, player_id in enumerate(active_ids):
            player_info = self.active_players[player_id]
            
            # Get average features for this player
            if player_id in self.player_features and len(self.player_features[player_id]) > 0:
                avg_features = np.mean(self.player_features[player_id], axis=0)
            else:
                continue
                
            for j, detection in enumerate(detections):
                # Calculate similarity
                similarity = self.calculate_feature_similarity(
                    avg_features,
                    detection['features'],
                    player_info['centroid'],
                    detection['centroid']
                )
                
                # Convert similarity to cost
                cost_matrix[i, j] = 1.0 - similarity
        
        # Solve assignment problem
        if n_players > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process matches
            matched_detections = set()
            
            for i, j in zip(row_indices, col_indices):
                if cost_matrix[i, j] < 0.6:  # Similarity > 0.4
                    player_id = active_ids[i]
                    detection = detections[j]
                    
                    # Update player
                    self.update_player_info(
                        player_id,
                        detection['centroid'],
                        detection['bbox'],
                        detection['features'],
                        detection['confidence'],
                        detection['quality_score']
                    )
                    
                    matched_detections.add(j)
            
            # Handle unmatched detections
            for j, detection in enumerate(detections):
                if j not in matched_detections:
                    # Try to match with disappeared players
                    matched_id = self.match_with_disappeared_players(detection)
                    
                    if matched_id is not None:
                        # Revive disappeared player
                        self.active_players[matched_id] = self.disappeared_players[matched_id]
                        self.update_player_info(
                            matched_id,
                            detection['centroid'],
                            detection['bbox'],
                            detection['features'],
                            detection['confidence'],
                            detection['quality_score']
                        )
                        del self.disappeared_players[matched_id]
                    else:
                        # Register new player (with high confidence threshold)
                        if detection['confidence'] > 0.75:
                            self.register_new_player(
                                detection['centroid'],
                                detection['bbox'],
                                detection['features'],
                                detection['confidence'],
                                detection['quality_score']
                            )
            
            # Handle unmatched players
            for i, player_id in enumerate(active_ids):
                if i not in row_indices or cost_matrix[i, col_indices[list(row_indices).index(i)]] >= 0.6:
                    self.active_players[player_id]['disappeared'] += 1
                    
                    if self.active_players[player_id]['disappeared'] > self.max_disappeared:
                        self.disappeared_players[player_id] = self.active_players[player_id]
                        del self.active_players[player_id]
    
    def cleanup_disappeared_players(self):
        """
        Clean up old disappeared players
        """
        to_remove = []
        for player_id, player_info in self.disappeared_players.items():
            player_info['disappeared'] += 1
            if player_info['disappeared'] > self.max_disappeared * 2:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            del self.disappeared_players[player_id]
            if player_id in self.player_features:
                del self.player_features[player_id]
            if player_id in self.player_positions:
                del self.player_positions[player_id]
            if player_id in self.player_velocities:
                del self.player_velocities[player_id]
            if player_id in self.player_stability:
                del self.player_stability[player_id]
    
    def get_stable_players(self):
        """
        Get only stable players for display
        
        Returns:
            Dictionary of stable players
        """
        stable_players = {}
        for player_id, player_info in self.active_players.items():
            if self.should_display_player(player_id):
                stable_players[player_id] = player_info
        
        return stable_players

def process_video_advanced(video_path, model_path='best.pt', output_path='advanced_tracking_output.mp4'):
    """
    Process video with advanced tracking system
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_path: Path to output video
    """
    # Initialize tracker
    tracker = AdvancedPlayerTracker(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color palette for player IDs
    colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (0, 128, 255),   # Light Blue
        (255, 128, 128), # Light Red
        (128, 255, 128), # Light Green
        (128, 128, 255), # Light Blue
        (255, 255, 128), # Light Yellow
        (255, 128, 255), # Light Magenta
        (128, 255, 255)  # Light Cyan
    ]
    
    frame_count = 0
    print("Starting video processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0 or frame_count == 1:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Update tracker
        stable_players = tracker.update(frame)
        
        # Draw tracking results
        for player_id, player_info in stable_players.items():
            x1, y1, x2, y2 = player_info['bbox']
            color = colors[player_id % len(colors)]
            
            # Get stability for visual feedback
            stability = tracker.get_player_stability(player_id)
            
            # Draw bounding box with thickness based on stability
            thickness = max(1, int(stability * 4))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Prepare label
            label = f'Player {player_id}'
            confidence = player_info['confidence']
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame,
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0] + 10, int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label,
                       (int(x1) + 5, int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center = player_info['centroid']
            cv2.circle(frame, (int(center[0]), int(center[1])), 4, color, -1)
            
            # Draw stability indicator (small bar)
            bar_width = 30
            bar_height = 4
            bar_x = int(x1)
            bar_y = int(y2) + 5
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Stability bar
            stable_width = int(bar_width * stability)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + stable_width, bar_y + bar_height), color, -1)
        
        # Draw frame information
        info_text = f'Frame: {frame_count} | Players: {len(stable_players)} | Total IDs: {tracker.next_player_id}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Unique players tracked: {tracker.next_player_id}")


if __name__ == "__main__":
    # Process the video
    video_path = "15sec_input_720p.mp4"
    model_path = "best.pt"
    output_path = "tracked_output.mp4"
    
    if os.path.exists(video_path) and os.path.exists(model_path):
        process_video_advanced(video_path, model_path, output_path)
    else:
        print("Please ensure both video file and model file exist:")
        print(f"Video: {video_path}")
        print(f"Model: {model_path}")
        
        print("\nTo use this code:")
        print("1. Place your video file '15sec_input_72Bp.mp4' in the same directory")
        print("2. Place your model file 'best.pt' in the same directory")
        print("3. Run the script")