# 💰 Census Income Classification

[![Python CI/CD](https://github.com/levisstrauss/census-income-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/levisstrauss/census-income-classifier/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning pipeline that predicts whether an individual's annual income exceeds $50,000 based on census data. Features a FastAPI backend with CI/CD and a Streamlit frontend.

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| **Frontend** | [Streamlit App](https://levisstrauss-census-income-classifier-frontendapp-lefxu3.streamlit.app/) |
| **API Docs** | [FastAPI Swagger](https://census-income-classifier.onrender.com/docs) |
| **API** | [Render Deployment](https://census-income-classifier.onrender.com/) |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Precision | 79.72% |
| Recall | 54.04% |
| F1 Score | 64.42% |

## 🚀 Features

- **ML Pipeline**: Random Forest classifier with train/test split
- **REST API**: FastAPI with GET and POST endpoints
- **CI/CD**: GitHub Actions (pytest + flake8) → Render auto-deploy
- **Slice Analysis**: Performance metrics across demographic groups
- **Frontend**: Interactive Streamlit web interface

## 📁 Project Structure
```
census-income-classifier/
├── ml/
│   ├── data.py          # Data processing
│   └── model.py         # Model training & inference
├── tests/
│   ├── test_model.py    # ML unit tests
│   └── test_api.py      # API unit tests
├── frontend/
│   └── app.py           # Streamlit frontend
├── main.py              # FastAPI application
├── train_model.py       # Training script
├── model_card.md        # Model documentation
└── slice_output.txt     # Slice performance metrics
```

## ⚡ Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/census-income-classifier.git
cd census-income-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run API locally
uvicorn main:app --reload

# Run tests
pytest tests/ -v
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/predict` | Income prediction |
| GET | `/docs` | Swagger documentation |

### Example Request
```bash
curl -X POST "https://your-render-url.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "workclass": "Private",
    "fnlgt": 200000,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 55,
    "native-country": "United-States"
  }'
```

### Response
```json
{
  "prediction": ">50K"
}
```

## 🛠️ Tech Stack

- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit
- **CI/CD**: GitHub Actions, Render
- **Testing**: pytest, flake8
