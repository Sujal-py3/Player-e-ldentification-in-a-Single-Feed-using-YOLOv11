# ⚽ Soccer Player Re-Identification Assignment

This repository contains the complete source code and documentation for the **Soccer Player Re-Identification** assignment using computer vision techniques. The system detects, tracks, and re-identifies soccer players from video input.

---

## 📁 Folder Structure

.
├── src/ # All source code
│ ├── yolov8_infer.py # YOLO detection module
│ ├── tracker.py # Deep SORT or custom tracker logic
│ ├── reid.py # Optional: Re-identification embeddings
│ └── utils.py # Helper functions
├── outputs/ # Output video and player trajectory JSON
│ ├── video_with_ids.mp4
│ └── player_tracks.json
├── input/ # Input video(s)
├── main.py # Main pipeline script
├── requirements.txt # Project dependencies
├── README.md # You're reading it!
└── report.md # Technical approach & methodology

yaml
Copy
Edit

---

## 🚀 How to Run

### 🧰 Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)
- Recommended: virtual environment (`venv`)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure ffmpeg is installed and added to your system PATH (for video output support).

▶️ Running the Code
Put your video (e.g., 15sec_input_720p.mp4) inside the root directory or input/, then run:

bash
Copy
Edit
python main.py
Outputs will be saved to the outputs/ folder:

video_with_ids.mp4: Tracked video with player IDs

player_tracks.json: Frame-by-frame player tracking data

🧠 Methodology Overview (see report.md for details)
✅ Detection + Tracking Pipeline
YOLOv8 for real-time person detection

Deep SORT (or custom tracker) for associating player IDs over time

Hungarian Algorithm for optimal assignment

Optionally used KMeans color clustering to assist with jersey-based ID refinement

🔍 Techniques Explored
Technique	Outcome
YOLOv8 + IOU/centroid tracking	Fast, but frequent ID mismatches
Deep SORT with cosine distance	Better long-term identity retention
Color histogram + KMeans	Minor improvement on similar uniforms
Custom feature embeddings	Promising but computationally heavier

🧗 Challenges Faced
Frequent occlusion & crowding disrupted ID continuity

Players with similar uniforms were hard to distinguish

Re-ID accuracy suffered when bounding boxes were jittery

Limited time to train task-specific re-identification model

📌 What’s Missing / Future Work
If given more time/resources, here’s how this system can be extended:

📸 OCR jersey number detection

🧠 Train a domain-specific Re-ID network

🎥 Integrate multi-camera view tracking

🚀 Optimize inference with TensorRT or ONNX

🧵 Add temporal smoothing for improved ID stability

🧪 Evaluation Criteria Checklist
✅ Criteria	Status
Player Re-Identification Accuracy	✅ Moderate with Deep SORT
Code Modularity & Clarity	✅ All modules in src/
Documentation Quality	✅ This README & report.md included
Runtime Efficiency	✅ Works on CPU/GPU, optional batch mode
Creativity in Approach	✅ Tried multiple matching strategies

🙋 Author
Sujal Singh
📍 B.Tech CSE, Bennett University
🔗 GitHub: Sujal-py3
📫 Email: sujal3177@gmail.com

🔒 Note: This submission is self-contained and ready to run. No external modifications required — just follow the instructions and you're good to go! 💪

python
Copy
Edit

---

Let me know if you'd like a fancy `report.md` to match this vibe, or if you want this converted into 
