# Minimum Viable Product (MVP) Specification
# B2B Procurement Recommender System

## 1. MVP Definition

The MVP is a **working prototype** that demonstrates the core value proposition: given a buyer institution's INN (tax identification number), the system returns a ranked list of food product categories the buyer is most likely to need, powered by a Two-Tower Neural Network.

### What the MVP IS:
- A trained Two-Tower model that produces buyer and product embeddings
- A FastAPI backend serving recommendations via REST API
- A web-based frontend for interactive exploration
- Evaluation results comparing the model against baselines
- A foundation for future MARKER platform integration

### What the MVP is NOT:
- A production-ready system (no authentication, no monitoring dashboards)
- A real-time learning system (model is trained offline)
- A supplier recommendation tool (buyer-side only in v1)

---

## 2. MVP Components

### Component 1: Data Pipeline (`backend/data/preprocessing.py`)

**What it does**: Takes the raw merged procurement dataset (Excel) and transforms it into features the neural network can learn from.

**Key functions**:
- `load_and_clean()` - Loads 9,994 records, removes duplicates, handles missing values
- `extract_institution_type()` - Identifies if a buyer is a school, hospital, kindergarten, etc.
- `temporal_split()` - Splits data chronologically (train: pre-Dec, val/test: December)
- `FeatureEngineer` class - Builds feature vectors for buyers and products

**Inputs**: `b2b_procurement_merged_dataset.xlsx`
**Outputs**: Buyer features, product features, interaction lists

### Component 2: Two-Tower Model (`backend/models/two_tower.py`)

**What it does**: The neural network that learns to match buyers with products.

**Architecture**:
- Buyer Tower: Takes institutional features → produces 64-dimensional embedding
- Product Tower: Takes product features → produces 64-dimensional embedding
- Matching: Cosine similarity between embeddings = recommendation score

**Key classes**:
- `BuyerTower` - Neural network for encoding buyer institutions
- `ProductTower` - Neural network for encoding food products
- `TwoTowerModel` - Combined model with contrastive loss function
- `ProcurementDataset` - PyTorch dataset for training pairs

### Component 3: Baseline Models (`backend/models/baselines.py`)

**What it does**: Simple models for comparison to prove the neural network adds value.

- `PopularityBaseline` - Recommends the most frequently purchased products (same for everyone)
- `MatrixFactorizationALS` - Traditional collaborative filtering (learns from purchase history only)

### Component 4: Training Pipeline (`backend/models/trainer.py`)

**What it does**: Trains all models, evaluates them, and saves results.

**Key functions**:
- `train_two_tower()` - Trains with early stopping on validation Recall@10
- `evaluate_model()` - Leave-one-out evaluation with Recall@K, NDCG@K
- `cold_start_evaluation()` - Stratified evaluation by buyer interaction count
- `build_faiss_index()` - Creates fast similarity search index

**Outputs saved to `saved_models/`**:
- `best_model.pt` - Trained Two-Tower model weights
- `feature_engineer.pkl` - Fitted feature transformers
- `product_embeddings.npy` - Precomputed product vectors
- `product_index.faiss` - FAISS index for fast retrieval
- `evaluation_results.json` - All metrics
- `webapp_artifacts.pkl` - Everything the API needs

### Component 5: FastAPI Backend (`backend/app/main.py`)

**What it does**: REST API that serves recommendations.

**Endpoints**:
| Endpoint | What it returns |
|----------|----------------|
| `GET /recommend/{inn}` | Top-K product recommendations for a buyer |
| `GET /buyer/{inn}/profile` | Buyer's institution type, region, purchase history |
| `GET /product/{code}/info` | Product details, number of buyers |
| `GET /products` | Full product catalog |
| `GET /buyers` | Searchable buyer list |
| `GET /stats` | Model performance metrics |

### Component 6: Frontend (`frontend/`)

**What it does**: Web interface for exploring recommendations.

**Pages**:
- **Dashboard**: System overview with key statistics
- **Recommendations**: Enter a buyer INN, see ranked product suggestions
- **Buyers**: Browse and search institutional buyers
- **Products**: View the food product catalog
- **Model Performance**: Evaluation metrics for all models

---

## 3. How to Run the MVP

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
cd b2b-recommender/backend
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
cd b2b-recommender/backend
python -m models.trainer
```
This takes approximately 2-5 minutes and saves all artifacts to `saved_models/`.

### Step 3: Start the API Server
```bash
cd b2b-recommender/backend
python -m uvicorn app.main:app --reload --port 8000
```

### Step 4: Open the Frontend
Open `frontend/index.html` in a web browser. The frontend connects to the API at `http://localhost:8000`.

### Step 5: Try a Recommendation
In the Recommendations page, enter a buyer INN (e.g., `7821007633`) and click "Get Recommendations".

---

## 4. File Structure for GitHub

```
b2b-recommender/
|
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py                 # FastAPI application
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- preprocessing.py        # Data pipeline & feature engineering
|   |   |-- b2b_procurement_merged_dataset.xlsx  # Source data
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- two_tower.py            # Two-Tower Neural Network
|   |   |-- baselines.py            # Popularity & MF-ALS baselines
|   |   |-- trainer.py              # Training & evaluation pipeline
|   |
|   |-- saved_models/               # Generated after training (gitignored)
|   |   |-- best_model.pt
|   |   |-- feature_engineer.pkl
|   |   |-- product_embeddings.npy
|   |   |-- product_index.faiss
|   |   |-- evaluation_results.json
|   |   |-- webapp_artifacts.pkl
|   |   |-- training_history.json
|   |
|   |-- requirements.txt
|
|-- frontend/
|   |-- index.html                  # Main HTML page
|   |-- src/
|       |-- app.js                  # Frontend JavaScript
|       |-- styles/
|           |-- main.css            # Stylesheet
|
|-- PRD.md                          # Product Requirements Document
|-- MVP.md                          # This file
|-- README.md                       # GitHub README
|-- .gitignore
```

### Files to Upload to GitHub
Upload **everything** in the structure above EXCEPT:
- `saved_models/` directory (generated by training, too large for Git)
- `__pycache__/` directories
- The `.xlsx` dataset file (if it's proprietary/confidential)

---

## 5. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Two-Tower over single-tower | Enables precomputing product embeddings offline for fast serving |
| OKPD2 as product key (not KTRU) | OKPD2 is 100% complete; KTRU has 21.7% missing |
| Cosine similarity (not dot product) | L2-normalized embeddings make similarity scores interpretable [0,1] |
| In-batch negative sampling | Standard for Two-Tower models; avoids explicit negative generation |
| FAISS for retrieval | Sub-linear search time; scales to millions of products |
| Temporal split (not random) | Prevents data leakage; mimics real deployment scenario |
| Vanilla JS frontend (not React) | Simpler deployment; no build step; easy to understand |

---

## 6. Limitations and Next Steps

### Current Limitations
1. **Small dataset**: 9,994 records limits model learning capacity
2. **Product concentration**: Dairy dominance makes popularity baseline very competitive
3. **No supplier recommendations**: Only buyer-side in MVP
4. **Offline model**: No real-time learning from new interactions
5. **No authentication**: API is open (fine for prototype, not production)

### Recommended Next Steps
1. Acquire more procurement data (target: 100K+ records)
2. Implement supplier-side recommendations
3. Add user authentication and rate limiting
4. Deploy to cloud (e.g., Google Cloud Run, AWS)
5. Integrate with MARKER platform API
6. Implement A/B testing for model comparison
7. Add model retraining pipeline (weekly/monthly)
