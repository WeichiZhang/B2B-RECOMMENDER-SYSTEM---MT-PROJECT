# B2B Procurement Recommender System

A Two-Tower Neural Network-based recommendation system for food procurement in the Russian B2B market. Built as a Master Thesis project for HSE Graduate School of Business.

## Overview

This system recommends food product categories to institutional buyers (schools, kindergartens, hospitals) based on their organizational profile, using a deep learning approach that handles the extreme data sparsity characteristic of B2B procurement.

**Architecture**: Two-Tower (Dual-Encoder) Neural Network
- **Buyer Tower**: Encodes institutional features (type, region, procurement methods, budget)
- **Product Tower**: Encodes product features (OKPD2 category, KTRU code, name, price)
- **Matching**: Cosine similarity in 64-dimensional shared embedding space

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd backend
python -m models.trainer
```

### 3. Start the API
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Open the Frontend
Open `frontend/index.html` in your browser.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /recommend/{inn}` | Get product recommendations for a buyer |
| `GET /buyer/{inn}/profile` | Get buyer profile |
| `GET /products` | List all product categories |
| `GET /buyers` | Search buyer institutions |
| `GET /stats` | Model performance metrics |

## Dataset

- **9,994** procurement records from Russia's MARKER platform
- **1,811** unique buyer institutions
- **72** OKPD2 food product categories
- **67** delivery regions across Russia

## Model Performance

| Model | Recall@5 | Recall@10 | NDCG@10 |
|-------|----------|-----------|---------|
| Two-Tower NN | 7.8% | 7.8% | 4.6% |
| Popularity Baseline | 83.1% | 92.2% | 45.6% |
| MF-ALS | 13.0% | 16.9% | 8.3% |

## Tech Stack

- **ML**: PyTorch, scikit-learn, FAISS
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript
- **Data**: pandas, openpyxl

## Project Structure

```
b2b-recommender/
├── backend/
│   ├── app/main.py              # FastAPI application
│   ├── data/preprocessing.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── two_tower.py         # Two-Tower Neural Network
│   │   ├── baselines.py         # Popularity & MF-ALS baselines
│   │   └── trainer.py           # Training & evaluation
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   └── src/
│       ├── app.js
│       └── styles/main.css
├── PRD.md
├── MVP.md
└── README.md
```

## Author

Weichi Zhang - HSE Graduate School of Business, Master of Business Analytics and Big Data Systems
