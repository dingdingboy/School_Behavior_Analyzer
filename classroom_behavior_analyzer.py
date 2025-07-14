import cv2
import numpy as np
import openvino as ov
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from collections import deque
from statistics import mean
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from model_handler import ModelHandler

@dataclass
class PersonCrop:
    """Data class to store person crop information"""
    frame: np.ndarray
    timestamp: datetime
    bbox: Tuple[int, int, int, int]

class PersonDetector:
    def __init__(self, model_path: str, device: str = "AUTO"):
        """Initialize person detector with OpenVINO model"""
        core = ov.Core()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        resized = cv2.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
        return input_data

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[List[float]]:
        """
        Detect persons in frame
        Returns: List of detections in format [[x, y, w, h, confidence], ...]
        """
        input_data = self.preprocess_image(frame)
        detections = self.compiled_model([input_data])[self.output_layer]
        
        results = []
        for detection in detections[0][0]:
            confidence = float(detection[2])
            if confidence > conf_threshold:
                # Get bbox coordinates
                xmin = max(0, int(detection[3] * frame.shape[1]))
                ymin = max(0, int(detection[4] * frame.shape[0]))
                xmax = min(frame.shape[1], int(detection[5] * frame.shape[1]))
                ymax = min(frame.shape[0], int(detection[6] * frame.shape[0]))
                
                # Calculate center point, width and height
                w = xmax - xmin
                h = ymax - ymin
                x = xmin  # right x
                y = ymin  # top y
                
                # Append detection in format [x_center, y_center, width, height, confidence]
                results.append([x, y, w, h, confidence])
        
        return results

class PersonTracker:
    def __init__(self, model_path: str, device: str = "AUTO"):
        """Initialize person tracking system"""
        self.detector = PersonDetector(model_path, device)
        self.frame_count = 0
        self.skip_frames = 30  # Process every 30th frame
        
        # Initialize DeepSORT with default feature extractor
        self.tracker = DeepSort(
            max_age=3,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.last_tracks = None
        self.person_crops = defaultdict(list)  # Track ID -> List[PersonCrop]
        self.max_crops_per_person = 3000  # Limit stored crops per person

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        self.frame_count += 1
        
        # Only detect on every 30th frame
        if self.frame_count % self.skip_frames == 0:
            # Detect persons
            detections = self.detector.detect(frame)
            
            if detections:
                # Update tracker with new detections
                prepared_detections = [(d[:4], d[4], "person") for d in detections]
                tracks = self.tracker.update_tracks(prepared_detections, frame=frame)
                
                if tracks:
                    self.last_tracks = tracks
        
        # Draw results using last known tracks
        if self.last_tracks:
            frame = self._draw_tracks(frame, self.last_tracks)
        
        return frame

    def _draw_tracks(self, frame, tracks):
        """Draw tracking results on frame and store person crops"""
        current_crops: Dict[int, PersonCrop] = {}
        clean_frame = frame.copy() # Copy the frame for crop, without rectangle boxes.
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            try:
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Store cropped person area
                person_crop = clean_frame[y1:y2, x1:x2].copy()  # .copy() to avoid reference issues
                current_crops[track_id] = PersonCrop(
                    frame=person_crop,
                    timestamp=datetime.now(),
                    bbox=(x1, y1, x2, y2)
                )
                
                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                continue
        
        # Update crops collection with new data
        if self.frame_count % self.skip_frames == 0:
            for track_id, crop_data in current_crops.items():
                self.person_crops[track_id].append(crop_data)
                # Keep only last N crops per person
                if len(self.person_crops[track_id]) > self.max_crops_per_person:
                    self.person_crops[track_id].pop(0)  # Remove oldest crop
        
        return frame

    def get_person_crops(self, track_id: int) -> List[PersonCrop]:
        """Get stored crops for a specific track ID"""
        return self.person_crops.get(track_id, [])

    def get_all_person_crops(self) -> Dict[int, List[PersonCrop]]:
        """Get all stored person crops"""
        return self.person_crops

class VideoReader:
    """Efficient video file reader with built-in resizing"""
    def __init__(self, source, max_width=1920, queue_size=128):
        self.source = source
        self.max_width = max_width
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self
    
    def update(self):
        cap = cv2.VideoCapture(self.source)
        while not self.stopped:
            if not self.queue.full():
                ret, frame = cap.read()
                if not ret:
                    self.stopped = True
                    break
                    
                # Resize frame if needed
                frame_h, frame_w = frame.shape[:2]
                if frame_w > self.max_width:
                    scale = self.max_width / frame_w
                    new_size = (self.max_width, int(frame_h * scale))
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                    
                self.queue.put(frame)
            else:
                time.sleep(0.001)  # Prevent busy-waiting
        cap.release()
    
    def read(self):
        return self.queue.get()
    
    def running(self):
        return not self.stopped or not self.queue.empty()
    
    def stop(self):
        self.stopped = True

class VideoWindow:
    def __init__(self, title="Person Tracking"):
        self.root = tk.Tk()
        self.root.title(title)
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size (80% of screen size)
        self.window_width = int(screen_width * 0.8)
        self.window_height = int(screen_height * 0.8)
        
        # Calculate position for center of screen
        position_x = (screen_width - self.window_width) // 2
        position_y = (screen_height - self.window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{self.window_width}x{self.window_height}+{position_x}+{position_y}")
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame with grid
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        # Create video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew", pady=10)
        
        # Create control frame at bottom
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        # Create source selection frame
        self.source_frame = ttk.LabelFrame(self.control_frame, text="Video Source")
        self.source_frame.pack(side=tk.LEFT, padx=5, pady=5)

        # Create radio buttons for source selection
        self.source_var = tk.StringVar(value="camera")
        self.camera_radio = ttk.Radiobutton(
            self.source_frame,
            text="Camera",
            variable=self.source_var,
            value="camera",
            command=self.on_source_change
        )
        self.camera_radio.pack(side=tk.LEFT, padx=5)

        self.file_radio = ttk.Radiobutton(
            self.source_frame,
            text="Video File",
            variable=self.source_var,
            value="file",
            command=self.on_source_change
        )
        self.file_radio.pack(side=tk.LEFT, padx=5)

        # Create camera selection frame
        self.camera_frame = ttk.Frame(self.control_frame)
        self.camera_frame.pack(side=tk.LEFT, padx=5)
        
        # Add camera selection label and dropdown
        self.camera_label = ttk.Label(self.camera_frame, text="Select Camera:")
        self.camera_label.pack(side=tk.LEFT, padx=5)
        
        # Get available cameras
        self.available_cameras = self.get_available_cameras()
        
        # Create camera selection dropdown
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(
            self.camera_frame, 
            textvariable=self.camera_var,
            values=list(self.available_cameras.keys()),
            state="readonly",
            width=30
        )
        if self.available_cameras:
            self.camera_dropdown.set(list(self.available_cameras.keys())[0])
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)

        # Create file selection frame
        self.file_frame = ttk.Frame(self.control_frame)
        self.file_frame.pack(side=tk.LEFT, padx=5)
        self.file_frame.pack_forget()  # Initially hidden

        # Add file selection button
        self.file_path = tk.StringVar()
        self.file_button = ttk.Button(
            self.file_frame,
            text="Choose Video File",
            command=self.choose_file
        )
        self.file_button.pack(side=tk.LEFT, padx=5)

        # Add file path label
        self.file_label = ttk.Label(self.file_frame, textvariable=self.file_path)
        self.file_label.pack(side=tk.LEFT, padx=5)

        # Add loop checkbox for video file
        self.loop_var = tk.BooleanVar(value=True)
        self.loop_check = ttk.Checkbutton(
            self.file_frame,
            text="Loop Video",
            variable=self.loop_var
        )
        self.loop_check.pack(side=tk.LEFT, padx=5)

        # Add save crops option
        self.save_crops_var = tk.BooleanVar(value=False)
        self.save_crops_cb = ttk.Checkbutton(
            self.control_frame,
            text="Save person crops as images",
            variable=self.save_crops_var
        )
        self.save_crops_cb.pack(side=tk.LEFT, padx=5)

        # Create buttons
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_tracking)
        self.start_button.pack(side=tk.LEFT, padx=5)
        # Add Analyze button
        self.analyze_button = ttk.Button(self.control_frame, text="Analyze", command=self.start_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        self.analyze_button.config(state='disabled')

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_tracking)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state='disabled')
        
        # Initialize video capture and tracking variables
        self.cap = None
        self.tracker = None
        self.is_tracking = False
        
        # Initialize FPS and profiling variables
        self.fps_times = deque(maxlen=30)  # Store last 30 frame times
        self.profiling_stats = {
            'read': deque(maxlen=30),
            'resize': deque(maxlen=30),
            'detect': deque(maxlen=30),
            'convert': deque(maxlen=30),
            'display': deque(maxlen=30)
        }
        self.show_stats = True  # Toggle stats display
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_resize)

        # Add analysis frame for behavior analysis
        self.analysis_frame = ttk.LabelFrame(self.main_frame, text="Behavior Analysis")
        self.analysis_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Create text widget for analysis results
        self.analysis_text = tk.Text(self.analysis_frame, width=40, height=30)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbar
        analysis_scrollbar = ttk.Scrollbar(self.analysis_frame, orient="vertical", command=self.analysis_text.yview)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)

    def get_available_cameras(self) -> dict:
        """Get all available cameras"""
        available_cameras = {}
        
        # Try the first 10 camera indices
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow
            if cap.isOpened():
                # Get camera name if possible
                cap.set(cv2.CAP_PROP_SETTINGS, 1)  # Try to open settings
                name = f"Camera {i}"
                
                # Try to get resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width and height:
                    name += f" ({width}x{height})"
                
                available_cameras[name] = i
                cap.release()
        
        return available_cameras
    
    def on_source_change(self):
        """Handle video source change"""
        if self.source_var.get() == "camera":
            self.file_frame.pack_forget()
            self.camera_frame.pack(side=tk.LEFT, padx=5)
            self.camera_frame.pack_forget()
            self.file_frame.pack(side=tk.LEFT, padx=5)

    def choose_file(self):
        """Open file dialog to choose video file"""
        from tkinter import filedialog
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mkv'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Open a video file',
            filetypes=filetypes
        )
        if filename:
            self.file_path.set(filename)

    def start_tracking(self):
        if not self.is_tracking:
            if self.source_var.get() == "camera":
                selected_camera = self.camera_var.get()
                if not selected_camera:
                    return
                camera_index = self.available_cameras[selected_camera]
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                video_path = self.file_path.get()
                if not video_path:
                    return
                # Initialize VideoReader with max width parameter
                self.video_reader = VideoReader(video_path, max_width=1920).start()
        
            # Initialize tracker
            model_path = r"..\Video_Collab_MuliModal_AI\models\cv\intel\person-detection-0202\FP16\person-detection-0202.xml"
            self.tracker = PersonTracker(model_path, device="GPU")
            
            self.is_tracking = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.camera_dropdown.config(state='disabled')
            self.file_button.config(state='disabled')
            self.update_frame()

    def stop_tracking(self):
        """Stop tracking and analyze stored crops"""
        if self.tracker and self.save_crops_var.get():
            # Create base output directory
            output_base = Path("person_crops")
            output_base.mkdir(exist_ok=True)
            
            # Save crops for each person in their own subfolder
            all_crops = self.tracker.get_all_person_crops()
            for track_id, crops in all_crops.items():
                if crops:
                    person_dir = output_base / f"person_{track_id}"
                    if save_person_crops_to_video(crops, person_dir):
                        print(f"Saved {len(crops)} frames as video for person {track_id} to {person_dir}")
                    else:
                        print(f"Failed to save frames as video for person {track_id}")
            
            # Release person_crops after saving
            self.tracker.person_crops.clear()

        # Original cleanup code
        self.is_tracking = False
        if self.cap is not None:
            self.cap.release()
        if self.video_reader is not None:
            self.video_reader.stop()
            self.video_reader = None
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.camera_dropdown.config(state='readonly')
        self.file_button.config(state='normal')
        self.analyze_button.config(state='normal')

    def on_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            # Update window dimensions
            self.window_width = event.width
            self.window_height = event.height
            
            # Calculate new display size for video
            # Use 90% of the window size while maintaining aspect ratio
            display_width = int(self.window_width * 0.9)
            display_height = int(self.window_height * 0.8)  # Leave room for controls
            self.display_size = (display_width, display_height)

    def update_frame(self):
        if self.is_tracking:
            frame_start = time.time()
            
            # Measure frame read time
            t0 = time.time()
            if self.source_var.get() == "camera":
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_tracking()
                    return
                
                # Only resize camera frames here
                frame_h, frame_w = frame.shape[:2]
                if frame_w > 1920:
                    scale = 1920 / frame_w
                    new_size = (1920, int(frame_h * scale))
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                    t1 = time.time()
                    self.profiling_stats['resize'].append((t1 - t0) * 1000)
            else:
                if self.video_reader.running():
                    frame = self.video_reader.read()  # Already resized in VideoReader
                else:
                    if self.loop_var.get():
                        # Restart video
                        self.video_reader.stop()
                        self.video_reader = VideoReader(self.file_path.get()).start()
                        frame = self.video_reader.read()
                    else:
                        self.stop_tracking()
                        return
                        
            self.profiling_stats['read'].append((time.time() - t0) * 1000)
            
            # Measure detection/tracking time
            t0 = time.time()
            processed_frame = self.tracker.process_frame(frame)
            self.profiling_stats['detect'].append((time.time() - t0) * 1000)
            
            # Measure conversion time
            t0 = time.time()
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image.thumbnail(self.display_size, Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(image=pil_image)
            self.profiling_stats['convert'].append((time.time() - t0) * 1000)

            # Measure display time
            t0 = time.time()
            self.video_label.configure(image=self.photo)
            
            # Add profiling text overlay
            if self.show_stats:
                fps = 1.0 / (time.time() - frame_start+ 1e-6)
                self.fps_times.append(fps)
                avg_fps = mean(self.fps_times)
                
                stats_text = f"FPS: {avg_fps:.1f}\n"
                stats_text += f"Read: {mean(self.profiling_stats['read']):.1f}ms\n"
                
                # Only show resize stats for camera source
                if self.source_var.get() == "camera" and self.profiling_stats['resize']:
                    stats_text += f"Resize: {mean(self.profiling_stats['resize']):.1f}ms\n"
                    
                stats_text += f"Detect: {mean(self.profiling_stats['detect']):.1f}ms\n"
                stats_text += f"Convert: {mean(self.profiling_stats['convert']):.1f}ms\n"
                
                # Add stats label if not exists
                if not hasattr(self, 'stats_label'):
                    self.stats_label = ttk.Label(self.video_frame, 
                                               text=stats_text,
                                               background='black',
                                               foreground='white')
                    self.stats_label.grid(row=0, column=1, sticky='ne', padx=5, pady=5)
                else:
                    self.stats_label.configure(text=stats_text)
                    
            self.profiling_stats['display'].append((time.time() - t0) * 1000)

            # Calculate next frame delay
            frame_time = (time.time() - frame_start) * 1000
            delay = max(1, int(1000/33 - frame_time))
            self.root.after(delay, self.update_frame)

    def run(self):
        self.root.mainloop()
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        if self.is_tracking:
            self.stop_tracking()
        self.root.destroy()
    
    def model_load_status(self, status_str: str):
        """Wait until model is loaded successfully based on status string."""
        if "Successfully"  in status_str:
            self.model_readiness = True
    
    def start_analysis(self):
        """Start behavior analysis in a background thread to keep UI responsive."""
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Analyzing behaviors...\n\n")
        self.root.update()

        # Run analysis in a separate thread
        analysis_thread = threading.Thread(target=self.analyze_person_behavior_async, daemon=True)
        analysis_thread.start()

    def analyze_person_behavior_async(self):
        """Background thread for analyzing behavior for all persons using VLM by processing MP4 files."""
        # Model loading (runs in this thread, but doesn't block UI)
        if not hasattr(self, 'model_handler'):
            self.model_handler = ModelHandler()
            model_path = "..\\Video_Collab_MuliModal_AI\\models\\vlm\\Qwen2.5-VL-3B\\INT4"
            device = "GPU"
            load_thread = self.model_handler.load_model(model_path, device, self.model_load_status)
            load_thread.start()
            self.model_readiness = False
            while not self.model_readiness:
                print("Waiting for the model to load")
                time.sleep(1)

        output_base = Path("person_crops")
        if not output_base.exists():
            self.root.after(0, lambda: self.analysis_text.insert(tk.END, "No person crops found for analysis.\nAnalysis complete."))
            return

        for person_dir in output_base.iterdir():
            if person_dir.is_dir():
                track_id = person_dir.name.split('_')[-1]
                mp4_files = list(person_dir.glob('*.mp4'))
                if mp4_files:
                    mp4_path = mp4_files[0]
                    prompt = ("You are a human posture analyzer. Analyze the person posture in the video"
                              "focus on posture change between frames and tell if the person sitting down or standing up in the video."
                              "at the end, summarize the analysis report and answer yes or no")
                    try:
                        result, _, _ = self.model_handler.run_inference(str(mp4_path), prompt)
                        summary = f"\n=== Person {track_id} Analysis ===\n{result}\n"
                        self.root.after(0, lambda s=summary: self.analysis_text.insert(tk.END, s))
                    except Exception as e:
                        error_msg = f"\n=== Person {track_id} Analysis ===\nError: {str(e)}\n"
                        print(f"Error analyzing video for person {track_id}: {e}")
                        self.root.after(0, lambda s=error_msg: self.analysis_text.insert(tk.END, s))
        self.root.after(0, lambda: self.analysis_text.insert(tk.END, "\nAnalysis complete."))

def save_person_crops_to_video(crops: List[PersonCrop], output_dir: Path, fps: int = 30) -> bool:
    """Save person crops as MP4 video"""
    if not crops:
        return False

    if output_dir.exists() and output_dir.is_file():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "person_crops.mp4"

    first_crop = crops[0].frame
    h, w = first_crop.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    actual_fps = int(fps/30)
    out = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (w, h))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    try:
        for crop in crops:
            if crop.frame.shape[0] != h or crop.frame.shape[1] != w:
                resized = cv2.resize(crop.frame, (w, h))
                out.write(resized)
            else:
                out.write(crop.frame)
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False
    finally:
        out.release()

def save_person_crops_as_images(crops: List[PersonCrop], output_dir: Path) -> bool:
    """Save person crops as individual images"""
    if not crops:
        return False
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for crop in crops:
            # Format timestamp for filename
            timestamp = crop.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            output_path = output_dir / f"frame_{timestamp}.jpg"
            cv2.imwrite(str(output_path), crop.frame)
        return True
    except Exception as e:
        print(f"Error saving images: {e}")
        return False

def main():
    app = VideoWindow()
    app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.run()

if __name__ == "__main__":
    main()