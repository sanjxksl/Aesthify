# Aesthify

> **A Framework for Exploring Aesthetic Perception in Interior Layouts**  
> *Understanding how people experience design across cultural and demographic contexts*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Framework-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

**Aesthify** is a modular research framework for exploring how users perceive visual aesthetics in multi-object interior layouts. Rather than evaluating or predicting aesthetic quality, Aesthify reveals the inherently subjective nature of aesthetic judgment across different demographic and cultural contexts.

The framework combines **YOLOv8 object detection** with **rule-based design principle analysis** to quantify visual structure, then compares these measurements with actual user perceptions to uncover patterns of aesthetic subjectivity.

> Aesthify demonstrates that aesthetics cannot be objectively evaluated - it's a tool for researchers and designers to understand perception diversity.

---

## Research Findings

Based on a 100+ participant user study, key insights include:

- **Simplicity** shows strongest correlation with user preferences (r = 0.68)
- **Symmetry** negatively correlates with aesthetic ratings (r = -0.60)
- **Cultural and demographic factors** significantly influence aesthetic perception
- **Aesthetic judgment varies dramatically** even for structurally similar layouts

---

## Key Features

- **Object Detection**: YOLOv8 and Roboflow model integration
- **Design Principle Analysis**: Quantifies 7 visual principles using interpretable rules
- **Perception Research**: Compare computed scores with user responses
- **Cultural Analysis**: Demographic clustering and preference patterns
- **Web Interface**: Upload images and visualize results
- **Survey Integration**: Tools for conducting perception studies

### Design Principles Analyzed

- Balance - visual weight distribution
- Proportion - relative object sizing  
- Symmetry - axis-based alignment
- Simplicity - visual complexity measures
- Harmony - color consistency
- Contrast - visual differentiation
- Unity - gestalt-based grouping

---

## Tech Stack

| Component        | Tools                                 |
|------------------|----------------------------------------|
| Language         | Python 3.10+                           |
| Backend          | Flask                                  |
| Computer Vision  | OpenCV, YOLOv8, Roboflow API           |
| Data Analysis    | Pandas, Scikit-learn, OpenPyXL         |
| Frontend         | HTML (Jinja2), JavaScript, Bootstrap   |
| Visualization    | Matplotlib, Seaborn                    |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sanjxksl/aesthify.git
cd aesthify
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Unix/macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: Roboflow API Setup

```bash
# Create .env file
ROBOFLOW_API_KEY=your-api-key
```

---

## Running the Application

```bash
python app.py
```

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Basic Usage:
1. Upload an interior layout image
2. View detected objects and design principle scores
3. Analyze visual structure breakdown
4. Export results to Excel

---

## Project Structure

```
aesthify/
├── app.py                   # Flask application entry point
├── routes/                  # Route handlers
│   └── evaluation.py
├── utils/                   # Core logic modules
│   ├── aesthetic_scoring.py # Design principle calculations
│   ├── detection_pipeline.py # Object detection logic
│   └── image_utils.py       # Image processing utilities
├── templates/               # HTML templates
├── static/                  # CSS, JavaScript, assets
├── interior_analysis/       # Survey analysis tools
│   └── survey_analysis.py   # User study comparison logic
├── models/                  # Detection model storage
├── results/                 # Output files and logs
└── requirements.txt
```

---

## Research Mode: User Perception Analysis

For researchers conducting perception studies:

### Setup Survey Analysis

1. Process images through Aesthify to get computed scores
2. Collect user ratings via survey (Google Forms template available)
3. Run comparative analysis:

```bash
python interior_analysis/survey_analysis.py
```

### Outputs Include:

- **Correlation Analysis**: How design principles align with user preferences
- **Demographic Clustering**: User groups based on aesthetic preferences  
- **Cultural Patterns**: Geographic and professional preference variations
- **Symbolic Mapping**: Emotional tags (calm, elegant, etc.) linked to visual features

---

## Example Analysis Results

| Design Principle | User Correlation | Interpretation |
|-----------------|------------------|----------------|
| Simplicity | +0.68 | Strong positive preference |
| Contrast | +0.56 | Moderate positive preference |
| Symmetry | -0.60 | Users prefer asymmetric layouts |
| Unity | +0.40 | Weak positive preference |

*Results show aesthetic perception varies significantly across demographics and cultures.*

---

## Deployment

### Local Development
```bash
python app.py
```

### Production
```bash
gunicorn app:app
```

Supports deployment on Render, Heroku, AWS EB with included configuration files.

---

## Use Cases

- **Design Research**: Understanding perception patterns across user groups
- **Cultural Studies**: Analyzing aesthetic preferences across demographics  
- **Education**: Teaching design principles through measurable examples
- **User Experience**: Exploring how visual structure affects user responses

---

## Academic Context

This framework emerged from research at IIITDM Kancheepuram exploring the subjective nature of aesthetic judgment. The work demonstrates that:

- Visual aesthetics cannot be objectively measured or predicted
- Cultural and demographic factors significantly influence perception
- Rule-based analysis reveals structural patterns but not universal preferences
- User studies are essential for understanding design perception

---

## Limitations

- Limited to 2D interior layout analysis
- Rule-based scoring may not capture all perceptual factors
- Detection accuracy varies with image quality and object arrangement
- Cultural findings may not generalize beyond study demographics

---

## Future Development

- 3D scene analysis capability
- Extended domain support (web design, product layouts)
- Machine learning adaptation from user feedback
- Cross-cultural perception studies
- Real-time design feedback integration

---

## Contributing

This is a research framework. For collaboration or extension:

1. Fork the repository
2. Create feature branches for new analysis methods
3. Maintain modular structure for easy testing
4. Document new perception research findings

---

## Author

**K S L Sanjana** (ME21B1015)  
Master of Management Analytics Candidate, Rotman School of Management  
[LinkedIn](https://linkedin.com/in/sanjanaksl) • [GitHub](https://github.com/sanjxksl)

---

## License

Academic and research use. Contact author for commercial applications.

---

## References

- Hu, H., et al. (2022). "A quantitative aesthetic measurement method for product appearance design"
- YOLOv8 by Ultralytics
- Roboflow Computer Vision Platform

---

## Acknowledgments

Special thanks to Dr. Sudhir Varadarajan (IIITDM Kancheepuram) for project supervision and the 101 participants who contributed to the user perception study.

> *"Aesthify reveals that beauty truly lies in the eye of the beholder - and those eyes are shaped by culture, context, and individual experience."*
