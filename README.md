<h2 align="center">Soccer Player Re-Identification Assignment</h2>

<p align="center">
  A computer vision pipeline to detect and re-identify players across video frames using YOLOv8 and tracking logic.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-orange?style=flat&logo=opencv" />
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-red?style=flat&logo=ai" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

---

## 📌 Problem Statement

Detect and consistently track soccer players in a video using object detection and re-identification techniques. The final output includes:

- A tracked video with unique player IDs
- A structured log file with player trajectories

---
MODEL = https://drive.google.com/file/d/1-5fOSHOSB9UXyP enOoZNAMScrePVcMD/view
## 🗂 Folder Structure



.
├── 15sec_input_720p.mp4 # Input soccer match video
├── best.pt # YOLOv8 model weights
├── tracked_output.mp4 # Video with player tracking annotations
├── output_final.mp4 # (Optional) Post-processed final output
├── tracking_script.py # Main tracking pipeline
---

## How to Run

### 1. Install dependencies

```bash
pip install ultralytics opencv-python numpy
```
### 2. Run the script
```bash
python tracking_script.py
```
---

### 3. Outputs
tracked_output.mp4: Tracked video with bounding boxes and unique player IDs

output_final.mp4: output video 

---
### Methodology
- Detection: YOLOv8 is used for per-frame player detection.

- Tracking: Basic ID assignment using centroid distance and IOU matching.

- Re-identification: Players are matched across frames based on spatial proximity.

- Output: Annotated video and optional log file for player positions over time.

---
## Experiments and Observations

- **Baseline**:
  - YOLOv8 + centroid-based tracking

- **Challenges**:
  - Player occlusion
  - Jersey similarity
  - Camera motion

- **Improvements Considered**:
  - Deep SORT integration
  - KMeans clustering on jersey colors
  - Re-ID embeddings (e.g., cosine distance, face features)

---
### Challenges
- Occlusion between players causes identity switching

- Visually similar players are difficult to distinguish

- Real-time processing is slower without GPU acceleration

---
### Future work
- Incorporate OCR for jersey number recognition

- Add appearance-based Re-ID for stronger identity consistency

- Implement multi-camera feed tracking

- Use ONNX or TensorRT for optimized inference
---
### Author
- Sujal Singh
- B.Tech CSE, Bennett University
- GitHub: @Sujal-py3
- Email: sujal3177@gmail.com
