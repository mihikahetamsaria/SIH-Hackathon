import cv2
import time
import os
import pickle
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Output Schema
# ---------------------------------------------------------
class FrameAnalysis(BaseModel):
    timestamp_sec: float = Field(description="The exact timestamp in seconds read from the image frame.")
    is_error: bool = Field(description="True if protocol violated or poor microgravity stabilization.")
    hardware_interaction: str = Field(description="Specific hardware handled (e.g., fluid valves, clamps).")
    action_description: str = Field(description="Kinematic description of the current action.")
    correction: str = Field(description="If is_error is True, specify the mechanical correction required.")

class ExperimentTimeline(BaseModel):
    timeline: list[FrameAnalysis]

# ---------------------------------------------------------
# 2. Phase 1: Chunked Pre-Compute Analysis
# ---------------------------------------------------------
def pre_analyze_video_chunked(video_path: str, protocol: str, target_fps: float = 3.0, chunk_size: int = 100) -> list[FrameAnalysis]:
    print(f"\n[PHASE 1] Extracting telemetry from {video_path} at {target_fps} FPS...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open {video_path}")
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    
    parts = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / original_fps
            
            # Burn timestamp into the frame
            cv2.putText(frame, f"T+{timestamp:.2f}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                parts.append({
                    "part": types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'),
                    "timestamp": timestamp
                })
                
        frame_count += 1
    cap.release()
    
    print(f"Extracted {len(parts)} total frames. Splitting into chunks of {chunk_size}...")
    
    client = genai.Client()
    master_timeline = []
    
    for i in range(0, len(parts), chunk_size):
        chunk = parts[i:i + chunk_size]
        chunk_parts = [item["part"] for item in chunk]
        start_t = chunk[0]["timestamp"]
        end_t = chunk[-1]["timestamp"]
        
        print(f"Processing chunk {i // chunk_size + 1} (Frames covering T+{start_t:.1f}s to T+{end_t:.1f}s)...")
        
        prompt_text = f"""
        Analyze this sequential batch of frames from a microgravity experiment chronologically.
        These frames span from approximately T+{start_t:.1f}s to T+{end_t:.1f}s. 
        The exact timestamp is burned into the top-left of every frame.
        
        Expected Protocol:
        {protocol}
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExperimentTimeline,
            temperature=0.0,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-3.7-flash',
                contents=[prompt_text] + chunk_parts,
                config=config
            )
            
            chunk_analysis: ExperimentTimeline = response.parsed
            master_timeline.extend(chunk_analysis.timeline)
            
        except Exception as e:
            print(f"⚠️ Error processing chunk starting at T+{start_t:.1f}s: {e}")

    sorted_timeline = sorted(master_timeline, key=lambda x: x.timestamp_sec)
    print(f"[PHASE 1 COMPLETE] Master timeline assembled with {len(sorted_timeline)} entries.")
    return sorted_timeline

# ---------------------------------------------------------
# 3. Phase 2: Tkinter UI Playback Application
# ---------------------------------------------------------
class TelemetryApp:
    def __init__(self, root, video_path, timeline):
        self.root = root
        self.root.title("Microgravity Experiment Telemetry Monitor")
        self.root.geometry("1100x650")
        self.root.configure(bg="#1e1e1e")
        
        self.video_path = video_path
        self.timeline = list(timeline)
        
        # --- Left Side: Video Display Canvas ---
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # --- Right Side: Telemetry Log Terminal ---
        self.log_frame = tk.Frame(root, bg="#252526")
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.log_frame, text="AI Telemetry Live Stream", fg="#ffffff", bg="#252526", font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=5)
        
        self.text_box = tk.Text(self.log_frame, bg="#1e1e1e", fg="#00ff00", insertbackground="white", font=("Courier", 11))
        self.text_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0
        
        self.is_playing = True
        self.thread = threading.Thread(target=self.playback_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_message(self, message):
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)

    def playback_loop(self):
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("\n--- Playback Concluded ---")
                break
                
            current_playback_time = self.frame_count / self.fps
            
            while self.timeline and current_playback_time >= self.timeline[0].timestamp_sec:
                telem = self.timeline.pop(0)
                status = "❌ ERROR" if telem.is_error else "✅ NOMINAL"
                self.log_message(f"[T+{telem.timestamp_sec:.2f}s] {status}")
                self.log_message(f"Hardware: {telem.hardware_interaction}")
                self.log_message(f"Action: {telem.action_description}")
                if telem.is_error:
                    self.log_message(f"Correction: {telem.correction}")
                self.log_message("-" * 40)
                
            cv2.putText(frame, f"T+{current_playback_time:.2f}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            canvas_width = self.canvas.winfo_width() or 600
            canvas_height = self.canvas.winfo_height() or 500
            if canvas_width > 10 and canvas_height > 10:
                image = image.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
                
            photo = ImageTk.PhotoImage(image=image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            
            self.frame_count += 1
            time.sleep(1.0 / self.fps)

    def on_close(self):
        self.is_playing = False
        self.cap.release()
        self.root.destroy()

# ---------------------------------------------------------
# Execution Entry Point with Caching
# ---------------------------------------------------------
if __name__ == "__main__":
    protocol = """
    1. Align the primary fluid coupling perfectly straight.
    2. Secure using the specialized torque driver.
    3. The astronaut must maintain three points of contact on the handrails.
    """
    video_file = "WhatsApp Video 2026-09-08 at 12.52.08 PM copy.mp4"
    cache_file = "timeline_cache.pkl"
    
    pre_computed_timeline = None
    
    # Check if cached results already exist to skip Phase 1 API calls
    if os.path.exists(cache_file):
        print(f"\n[CACHE] Found saved analysis file ({cache_file}). Skipping Phase 1 API calls!")
        with open(cache_file, "rb") as f:
            pre_computed_timeline = pickle.load(f)
    else:
        pre_computed_timeline = pre_analyze_video_chunked(
            video_path=video_file, 
            protocol=protocol, 
            target_fps=3.0, 
            chunk_size=100
        )
        if pre_computed_timeline:
            with open(cache_file, "wb") as f:
                pickle.dump(pre_computed_timeline, f)
            print(f"[CACHE] Timeline saved successfully to {cache_file}.")

    # Phase 2: Launch UI window
    if pre_computed_timeline:
        root = tk.Tk()
        app = TelemetryApp(root, video_file, pre_computed_timeline)
        root.mainloop()
    else:
        print("Error: No timeline data available.")