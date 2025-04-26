# 🎨 Aesthtify

> **Quantitative Aesthetic Evaluation for Interior Design Layouts**  
> *Research-grade scoring engine for interior design analysis.*

---

## ✨ Overview

**Aesthtify** is a computer vision-powered framework designed to quantify and evaluate the aesthetic quality of interior design layouts.  
It blends design theory with modern image processing to deliver structured, data-driven insights into visual balance, harmony, and appeal.

🔎 **Focus Areas**:
- Symmetry, balance, harmony, simplicity, contrast, unity, proportion
- Object detection via YOLOv8 and Roboflow APIs
- Visual contour and edge structure analysis
- Survey-based validation against human perception (optional)

---

## 🛠️ Key Features

- Aesthetic scoring engine covering multiple design principles
- YOLOv8-based object detection with Roboflow integration
- Contour clustering and closure analysis
- Labeled layout visualizations with bounding boxes
- Survey-based scoring analysis and perception validation
- Structured exports: CSV reports, Excel dumps, labeled images
- Plotting modules: scatter plots, bar charts, aesthetic vs perception correlations
- Modular, editable Python package with `.env` configurations
- Clean Flask web interface for easy layout evaluations

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
git clone https://github.com/yourusername/aesthtify.git
cd aesthtify
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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

### 📝 Survey Evaluation and Research Mode

For academic perception vs algorithmic analysis:

```bash
python -m interior_analysis.survey_analysis
```

- Loads `.xlsx` survey responses.
- Evaluates corresponding layout images.
- Outputs:
  - CSV files in `/results/csv_outputs/`
  - Statistical plots in `/results/plots/`
  - Learned optimal aesthetic weights
  - Perception vs Computed Score correlations

---

## 📂 Project Structure

```plaintext
aesthtify/
├── app.py                     # Flask application entry
├── .env                        # Environment variables
├── requirements.txt            # Dependency list
├── setup.py                    # Installable package config
├── README.md                   # Project documentation
│
├── utils/                      # Core logic: scoring, image utilities, config
├── routes/                     # Flask route blueprints
├── interior_analysis/          # Survey evaluation and analysis pipeline
├── models/                     # YOLO model files
├── results/                    # Evaluation results, plots, csvs
│   ├── csv_outputs/
│   └── plots/
├── templates/                  # Frontend HTML (Jinja2)
└── static/                     # JavaScript, CSS, assets
```

---

## 📊 Example Output

| Metric                 | Score  |
| ----------------------- | ------ |
| Symmetry Score          | 0.81   |
| Simplicity Score        | 0.90   |
| Unity Score             | 0.86   |
| Contrast Score          | 0.79   |
| Harmony Score           | 0.75   |
| Final Aesthetic Score   | 0.83   |

Additional outputs:
- Labeled layout images
- CSV datasets of perception vs algorithmic scores
- Scatter plots and correlation graphs
- Best-fit aesthetic weight discovery plots

---

## 👤 Author

**K S L Sanjana**  
[LinkedIn](https://linkedin.com/in/sanjanaksl) • [Email](mailto:sanjxksl@gmail.com)

---

## 📄 License

This project is made available for **academic reference purposes** only, under the guidelines of IIITDM Kancheepuram.  
Commercial usage, reproduction, or distribution requires **explicit written permission** from the author.

For inquiries, please contact: [sanjxksl@gmail.com]

---

## 🙏 Acknowledgements

- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- scikit-learn, OpenCV, Pandas libraries
- Foundational work in design theory and visual perception

---