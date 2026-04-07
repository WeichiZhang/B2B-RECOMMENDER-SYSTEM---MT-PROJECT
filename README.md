# B2B Procurement Recommender System

A Two-Tower Neural Network-based recommendation system for food procurement in the Russian B2B market. Built as a Master Thesis project for HSE Graduate School of Business.

## Overview

This system helps institutional buyers (schools, kindergartens, hospitals) discover reliable food suppliers by analyzing real procurement data from Russia's MARKER platform (Interfax Group).

**Key Features:**
- **Find Suppliers**: Select your institution type, region, and product → see ranked suppliers with prices, trust scores, and delivery cities
- **Compare Prices**: See how prices for any product vary across Russian regions
- **Peer Analysis**: Learn what products similar institutions in your region are buying
- **Data Analysis**: Interactive EDA charts from the procurement dataset
- **Model Performance**: Evaluation metrics for the Two-Tower NN vs. baselines

**Architecture**: Two-Tower (Dual-Encoder) Neural Network
- **Buyer Tower**: Encodes institutional features (type, region, procurement methods, budget)
- **Product Tower**: Encodes product features (OKPD2 category, KTRU code, name, price)
- **Matching**: Cosine similarity in 64-dimensional shared embedding space

## Quick Start

### Option 1: Standalone App (Recommended)
Simply open `B2B_Recommender_App.html` in any web browser. No installation needed — all data is embedded in the file.

### Option 2: Full Setup (Model Training + API)

#### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### 2. Train the Model
```bash
cd backend
python -m models.trainer
```

#### 3. Start the API
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

#### 4. Open the Frontend
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
- **371** suppliers
- **67** delivery regions across Russia

## Model Performance

### Overall
| Model | Recall@5 | Recall@10 | NDCG@10 |
|-------|----------|-----------|---------|
| Two-Tower NN | 7.8% | 7.8% | 4.6% |
| Popularity Baseline | 83.1% | 92.2% | 45.6% |
| MF-ALS | 13.0% | 16.9% | 8.3% |

### Cold-Start (Buyers with < 5 interactions)
| Model | Recall@5 | Recall@10 | NDCG@10 |
|-------|----------|-----------|---------|
| Two-Tower NN | 35.3% | 35.3% | 20.7% |
| Popularity | 41.2% | 76.5% | 37.7% |
| MF-ALS | 47.1% | 64.7% | 32.9% |

The Two-Tower model demonstrates its value for cold-start buyers, leveraging institutional features rather than interaction history alone.

## Tech Stack

- **ML**: PyTorch, scikit-learn, FAISS
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript (standalone, no build step)
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
├── B2B_Recommender_App.html     # Standalone app (just open in browser!)
├── PRD.md
├── MVP.md
└── README.md
```

## Author

Weichi Zhang - HSE Graduate School of Business, Master of Business Analytics and Big Data Systems
