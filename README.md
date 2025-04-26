# 🎨 Aesthify

> **Quantitative Aesthetic Evaluation for Interior Design Layouts**  
> *Research-grade scoring engine for interior design analysis.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Framework-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Overview

**Aesthify** is a computer vision-powered framework designed to quantify and evaluates the visual aesthetics of images, especially focused on room layouts and interior designs.  
It uses a combination of deep learning (YOLOv8 + Roboflow) for object detection and a handcrafted quantitative model based on Gestalt principles to score images across seven aesthetic dimensions.

🔎 **Focus Areas**:
- Symmetry, balance, harmony, simplicity, contrast, unity, proportion
- Object detection via YOLOv8 and Roboflow APIs
- Visual contour and edge structure analysis
- Survey-based validation against human perception (optional), performed for Interior designs, can be applied to different use cases

---

## 🛠️ Key Features

- 📷 Upload or capture images directly via the web interface
- 🧠 Object detection via YOLO and Roboflow models
- 🖼️ Aesthetic scoring on:
  - Balance
  - Proportion
  - Symmetry
  - Simplicity
  - Unity
  - Contrast
  - Harmony
- 📈 Visual analysis and annotation
- 📝 Survey data analysis and user clustering (optional mode)

---

## 🧪 Tech Stack

| Component        | Tools                                   |
| ---------------- | --------------------------------------- |
| Language         | Python 3.10+                            |
| Backend          | Flask                                   |
| Computer Vision  | OpenCV, YOLOv8 (Ultralytics)             |
| Data Processing  | Pandas, OpenPyXL, Scikit-learn           |
| Frontend         | HTML (Jinja2), JavaScript, Bootstrap     |
| Plotting         | Matplotlib, Seaborn                     |
| Packaging        | `setup.py`, `.env`, editable install     |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sanjxksl/aesthify.git
cd aesthify
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
python -m venv venv
source venv/bin/activate    # On Unix
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File

Inside project root:

```env
ROBOFLOW_API_KEY=your-roboflow-api-key
```
✅ (Required only if Roboflow models are enabled.)

---

## 📂 Project Structure

```txt
aesthify/
├── app.py               # Flask app entrypoint
├── utils/                # Core processing utilities
│   ├── aesthetic_scoring.py
│   ├── detection_pipeline.py
│   ├── image_utils.py
│   ├── main_pipeline.py
│   └── __init__.py
├── routes/               # Flask routes
│   ├── __init__.py
│   └── evaluation.py
├── static/               # Frontend assets
│   ├── styles.css
│   ├── script.js
│   └── logo.png
├── templates/            # HTML templates
│   └── index.html
├── models/               # (optional) YOLO models
├── results/              # Evaluation output files
├── .env                  # Environment variables
├── requirements.txt
├── setup.py
├── README.md
└── Procfile              # For production deployment
```

---

## 📈 Running the Application

### Web Application

```bash
python app.py
```

Open your browser:  
`http://127.0.0.1:5000`

Available actions:
- Upload a layout image
- View computed aesthetic scores
- Download structured results
- Visualize layout with detected object annotations

Results automatically saved to `results/evaluation_results_dump.xlsx`.

---

## 📊 Example Output

> Uploaded Image ➔ Detection ➔ Aesthetic Scores ➔ Labeled Image returned!

| Metric                 | Score  |
| ----------------------- | ------ |
| Symmetry Score          | 0.81   |
| Simplicity Score        | 0.90   |
| Unity Score             | 0.86   |
| Contrast Score          | 0.79   |
| Harmony Score           | 0.75   |
| Final Aesthetic Score   | 0.83   |

📷 Plus, get an annotated version of the image showing detected objects!

Additional outputs (when you run the interior_analysis/survey_analysis.py):
- CSV datasets of perception vs algorithmic scores
- Scatter plots and correlation graphs
- Best-fit aesthetic weight discovery plots

---

## 📚 Survey Mode (Optional)

If you have collected user feedback via a survey (Google Forms or similar),  
you can perform detailed analysis and cluster users based on their aesthetic perception.

---

### 📋 Steps for Survey Mode Setup:

1. **Evaluation Results Preparation**:
    - Open `evaluation_results_dump.xlsx`
    - Copy all **calculated evaluation scores**.
    - Map them correctly to their respective **Image IDs**.
    - Paste them into `evaluation_results.xlsx` under the correct columns.

2. **Survey Images Organization**:
    - Save all images shown in your survey into a folder named:
    ```
`
    survey_images/
    ```
    - Ensure filenames match the Image IDs used in your evaluation and survey.

3. **Image Mapping Update**:
    - Open `survey_analysis.py`
    - Update the `img_map` dictionary with your **new filenames** corresponding to survey IDs.

4. **Survey Results Insertion**:
    - Collect responses from your Google Form survey.
    - Download the Google Form results into `survey_results.xlsx`.
    - Ensure the format (columns/headers) matches expected fields in the analysis script.

    Example:
    📄 **Google Form Template Reference:**  
    [Click here to view the Survey Form](https://forms.gle/GEgJ71ow9mbKxjta9)

5. **Run Survey Analysis**:
```bash
python survey_analysis.py

---

## 🛰️ Deployment Notes

- Use Gunicorn + Procfile for cloud platforms (Heroku, Render, AWS Elastic Beanstalk)
- Recommended command:
```bash
gunicorn app:app
```

---

## 📂 Offline Installation & Operation

✅ Aesthify fully supports **offline operation** for sensitive environments.

Please refer to the detailed manual here:  
📄 [Aesthify Full Offline Setup and Operation Manual (PDF)](Aesthify_Full_Offline_Setup_and_Operation_Manual.pdf)

---

## 👤 Author

Developed by **K S L Sanjana**  
[LinkedIn](https://linkedin.com/in/sanjanaksl) • [Email](mailto:sanjxksl@gmail.com)

---

## 📄 License

This project is made available for **academic reference purposes** only, under the guidelines of IIITDM Kancheepuram.  
Commercial usage, reproduction, or distribution requires **explicit written permission** from the author.

For inquiries, please contact: [sanjxksl@gmail.com]

---

## 🙏 Acknowledgements

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [Roboflow Models](https://roboflow.com/)
- [Hu, Liu, Lu, Guo: A quantitative aesthetic measurement method for product appearance design (2022)](https://doi.org/10.1016/j.aei.2022.101644)

---
