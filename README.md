# ⚽ Soccer Player Re-Identification Assignment

A computer vision pipeline for detecting, tracking, and re-identifying soccer players in real-time video footage using YOLOv8 and Deep SORT.

> 🔍 Built as part of a technical assignment focused on real-world object tracking and ID management.

---

## 🎬 Demo

> *(Insert demo GIF or video link here if available)*  
> Example output: `outputs/video_with_ids.mp4`  
> Output JSON: `outputs/player_tracks.json`

---

## 🚀 Features

- 🧠 **YOLOv8-based player detection**
- 🔁 **Deep SORT tracking with cosine distance**
- 🧩 Modular structure with separate detection/tracking logic
- 🧵 Optional jersey color clustering via KMeans
- 💾 JSON export of all tracked player data
- 📹 Annotated video output with ID overlays

---

## 🗂️ Folder Structure


---

## 🧰 Installation & Setup

### 📦 Requirements

- Python 3.8 or higher
- `ffmpeg` installed and in PATH
- YOLOv8-compatible dependencies

### ⚙️ Setup Instructions

1. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows


```bash
pip install ultralytics opencv-python numpy

✅ ultralytics includes YOLOv8 support out of the box.

▶️ How to Run
Run the tracking script using:

```bash
python tracking_script.py
Make sure 15sec_input_720p.mp4 and best.pt are in the same folder.
Output will be saved as:

tracked_output.mp4 → Tracked player IDs on video

output_final.mp4 → (Optional final version)
---
### 🧠 Methodology
Detection: YOLOv8 with pretrained weights (best.pt)

Tracking: Simple ID assignment or distance-based re-identification

Visualization: OpenCV drawing functions to overlay IDs on frames

## 🔍 Techniques Explored
Technique	Outcome
YOLOv8 Detection	Fast and accurate
IOU-based tracking (simple)	Decent but may switch IDs on occlusion
Custom tweaks (optional)	Fine-tuned thresholding for smoother IDs

🧗 Challenges Faced
⚠️ Occlusion during close contact confused ID consistency

⌛ Re-identification without jersey number/OCR was limited

🐌 Some lag without GPU during real-time processing

🔮 Future Improvements
📸 Integrate OCR for jersey numbers

🧠 Add Deep SORT or appearance-based re-ID

🚀 Optimize runtime using ONNX/TensorRT

🧵 Add temporal smoothing and better ID switching logic

🙋‍♂️ **Author**
Sujal Singh
🎓 B.Tech CSE, Bennett University
🌐 GitHub: @Sujal-py3
📫 Email: sujal3177@gmail.com

