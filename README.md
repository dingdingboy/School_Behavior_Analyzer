# School Behavior Analyzer

A Python application for detecting, tracking, and analyzing classroom behavior using computer vision and large vision-language models (VLMs). The system detects and tracks people in video streams, saves cropped person videos, and analyzes posture changes using a VLM.

## Features

- **Person Detection & Tracking:** Uses OpenVINO and DeepSORT for real-time person detection and tracking in video streams or files.
- **Person Crop Extraction:** Automatically saves cropped video segments for each detected person.
- **Behavior Analysis:** Analyzes posture changes (e.g., sitting/standing) for each person using a VLM (Qwen2.5-VL-3B).
- **GUI:** User-friendly interface for selecting video sources, starting/stopping tracking, and viewing analysis results.

## Project Structure

```
classroom_behavior_analyzer.py   # Main GUI and tracking logic
model_handler.py                # VLM model loading and inference
person_crops/                   # Output directory for cropped person videos
```

## Requirements

- Python 3.8+
- OpenVINO
- OpenCV
- deep_sort_realtime
- transformers
- optimum
- Pillow
- tkinter

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

1. **Run the Application:**
   ```sh
   python classroom_behavior_analyzer.py
   ```

2. **Select Video Source:**
   - Choose between camera or video file.
   - Optionally enable "Save person crops as images".

3. **Start Tracking:**
   - Click "Start" to begin detection and tracking.
   - Cropped videos for each person will be saved in `person_crops/`.

4. **Analyze Behavior:**
   - After tracking, click "Analyze" to run posture analysis on each person's video.

## Output

- Cropped videos for each detected person are saved in `person_crops/person_{id}/person_crops.mp4`.
- Analysis results are displayed in the GUI and summarize posture changes.

## Notes

- The VLM model (Qwen2.5-VL-3B) must be downloaded and available at the specified path in the code.
- For best performance, use a machine with a compatible GPU and OpenVINO installed.

## License

This project is for research and educational purposes.

---

**Main files:**
- [`classroom_behavior_analyzer.py`](classroom_behavior_analyzer.py)
- [`model_handler.py`](model_handler.py)