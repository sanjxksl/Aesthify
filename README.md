# 🎨 Aesthify

> **Perception-Aware Aesthetic Analysis for Interior Layouts**  
> *An interpretable framework for exploring how people experience design.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Framework-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Overview

**Aesthify** is a perception-driven aesthetic analysis tool for evaluating multi-object interior layouts. Rather than predicting what looks good, Aesthify helps you explore how structured design principles—like *Symmetry, Balance, Simplicity,* and *Contrast*—are actually perceived by users.

It combines deep learning-based **object detection** (YOLOv8, Roboflow) with a **rule-based scoring engine** grounded in design theory. Each layout is evaluated across seven visual principles using explainable, quantitative logic.

💡 Aesthify is not a beauty predictor. It's a reflection tool for designers, researchers, and educators to interpret visual structure through the lens of perception.

---

### 🔍 Focus Areas

- Evaluate interior layouts using 7 design principles
- Detect objects using YOLOv8 or Roboflow models
- Score images via rule-based design logic (not ML)
- Visualize layout structure and perceptual alignment
- (Optional) Validate using user survey responses

---

## 🛠️ Key Features

- 📷 Upload interior images via web UI  
- 🧠 Object detection using YOLOv8 or Roboflow  
- 🖼️ Rule-based scoring on:
  - Simplicity
  - Balance
  - Symmetry
  - Unity
  - Proportion
  - Contrast
  - Harmony
- 📈 Visual annotation + score overlay  
- 🧪 Optional: user survey integration for clustering + correlation analysis

---

## 🧪 Tech Stack

| Component        | Tools                                 |
|------------------|----------------------------------------|
| Language         | Python 3.10+                           |
| Backend          | Flask                                  |
| Computer Vision  | OpenCV, YOLOv8, Roboflow API           |
| Data Processing  | Pandas, OpenPyXL, Scikit-learn         |
| Frontend         | HTML (Jinja2), JavaScript, Bootstrap   |
| Plotting         | Matplotlib, Seaborn                    |
| Packaging        | `setup.py`, `.env`, editable install   |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sanjxksl/aesthify.git
cd aesthify
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Unix
venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys (Optional)

If using Roboflow models, add:

```env
ROBOFLOW_API_KEY=your-roboflow-api-key
```

---

## 📂 Project Structure

```txt
aesthify/
├── app.py               # Flask entrypoint
├── utils/               # Aesthetic scoring + detection logic
├── routes/              # Route definitions
├── static/              # CSS, JS, and UI assets
├── templates/           # Jinja2 HTML templates
├── models/              # (optional) detection models
├── results/             # Output dump (scores, logs)
├── requirements.txt     # Dependencies
├── survey_analysis.py   # User survey & clustering analysis
├── README.md
└── .env                 # Roboflow API key (optional)
```

---

## 📈 Running the App

```bash
python app.py
```

Then open your browser:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

You can:

* Upload a layout image
* View principle-based aesthetic scores
* Visualize detected objects + layout structure
* Export all scores to Excel

---

## 📊 Example Output

| Metric                | Score |
| --------------------- | ----- |
| Simplicity            | 0.90  |
| Symmetry              | 0.81  |
| Unity                 | 0.86  |
| Contrast              | 0.79  |
| Harmony               | 0.75  |
| Final Aesthetic Score | 0.83  |

> Annotated layout + scores are saved in `/results`

---

## 📚 Survey Mode (Optional)

Aesthify supports user study integration for analyzing how real people perceive different layouts.

Use this mode to:

* Compare scores with actual user ratings
* Cluster users based on preference patterns
* Analyze symbolic tags (e.g., “cozy”, “elegant”) tied to visual features

---

### 📝 Survey Setup

1. Add computed scores to `evaluation_results.xlsx`
2. Add survey images to `survey_images/`
3. Update `img_map` in `survey_analysis.py`
4. Add your form responses to `survey_results.xlsx`
5. Run the analysis:

```bash
python interior_analysis.survey_analysis.py
```

Outputs:

* Perception vs. structure correlation plots
* Clustered user profiles
* Insights on how emotion tags map to layouts

---

## 🛰️ Deployment Notes

Use Gunicorn for production deployment:

```bash
gunicorn app:app
```

Supports Render, Heroku, or AWS EB with included `Procfile`.

---

## 🖥️ Offline Support

Aesthify is fully offline-capable for secure or demo environments.
Refer to the [Offline Setup Manual (PDF)](Aesthify_Full_Offline_Setup_and_Operation_Manual.pdf).

---

## 👩‍🎓 Author

Built by **K S L Sanjana**
[LinkedIn](https://linkedin.com/in/sanjanaksl) • [Email](mailto:sanjxksl@gmail.com)

---

## 📄 License

MIT License (Academic use only)

For research use and academic exploration.
Contact the author for commercial licensing or integration queries.

---

## Acknowledgements

* [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
* [Roboflow API + Universe Models](https://roboflow.com/)
* [Hu et al. (2022)](https://doi.org/10.1016/j.aei.2022.101644) – Aesthetic measurement in design

---

> 💬 *Aesthify doesn’t predict what looks good — it reflects how design feels to people.*
