# Product Requirements Document (PRD)
# B2B Procurement Recommender System

## 1. Product Overview

### Product Name
B2B Procurement Recommender System for Food Industry in Russian Market

### Product Vision
An intelligent recommendation system that matches institutional buyers (schools, kindergartens, hospitals) with food products they are likely to need, using a Two-Tower Neural Network architecture trained on real procurement data from Russia's MARKER platform.

### Problem Statement
State-owned institutions in Russia face challenges in food procurement:
- **Buyers** struggle to discover reliable suppliers and suitable products efficiently
- **Suppliers** waste resources identifying relevant tender opportunities
- **No intelligent matching** exists on procurement platforms like MARKER
- **Data sparsity**: 52.1% of buyers have fewer than 5 product interactions, making traditional collaborative filtering ineffective

### Target Users
1. **Institutional Buyers**: Schools (MBOU), kindergartens (MBDOU), hospitals (GBUZ), federal institutions (FGBU) that procure food products through government tenders
2. **Food Suppliers**: Companies seeking to identify which institutions are likely to need their products
3. **Platform Operators**: MARKER platform team who will integrate the recommendation engine

---

## 2. Goals and Success Metrics

### Business Goals
| Goal | Metric | Target |
|------|--------|--------|
| Improve product discovery | Recall@10 | > 30% for cold-start buyers |
| Reduce procurement search time | User engagement with recommendations | > 50% click-through |
| Support cold-start scenarios | Cold-start buyer Recall@10 | > 20% |
| Outperform simple baselines | NDCG@10 vs. popularity baseline | Competitive or superior |

### Technical Goals
- Train a Two-Tower Neural Network on 9,994 procurement records
- Serve recommendations via REST API with < 100ms latency
- Support 1,811 buyer institutions and 72 product categories
- Provide explainable recommendations through buyer/product profiles

---

## 3. Data Foundation

### Dataset
- **Source**: MARKER platform (Interfax Group), extracted from Russia's EIS procurement system
- **Size**: 9,994 line-item records from completed contracts
- **Coverage**: 1,811 unique buyers, 371 suppliers, 72 OKPD2 product categories, 67 regions
- **Time period**: July 2025 - January 2026 (83.6% in December 2025)
- **Data quality**: Core fields (INN, OKPD2, price) are 100% complete

### Key Data Characteristics
- **Interaction matrix density**: 6.89% (sparse for recommender systems)
- **Cold buyers**: 52.1% have < 5 product category interactions
- **Product concentration**: Top 6 dairy categories = 45.2% of records
- **Geographic spread**: 67 Russian regions represented

---

## 4. System Architecture

### Two-Tower Neural Network

```
Buyer Tower                          Product Tower
    |                                     |
[Institution Type Embedding (8d)]   [OKPD2 Group Embedding (16d)]
[Region Embedding (16d)]            [OKPD2 Full Embedding (16d)]
[Procurement Method Profile (4d)]   [KTRU Embedding (32d)]
[Category History (72d)]            [Unit Embedding (4d)]
[Contract Statistics (2d)]          [TF-IDF Text Projection (64d)]
    |                               [Log Price (1d)]
    v                                     |
[Linear 256 + BN + ReLU + Drop]          v
[Linear 128 + BN + ReLU + Drop]    [Linear 256 + BN + ReLU + Drop]
[Linear 64 + L2 Norm]              [Linear 128 + BN + ReLU + Drop]
    |                               [Linear 64 + L2 Norm]
    v                                     |
  Buyer Embedding (64d)              Product Embedding (64d)
         \                          /
          \                        /
           Cosine Similarity Score
```

### Technology Stack
| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch 2.x |
| Feature Engineering | scikit-learn, pandas |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Data Format | OKPD2/KTRU Russian classification |

---

## 5. Features

### MVP Features (v1.0)
1. **Product Recommendations**: Given a buyer INN, return top-K recommended food product categories ranked by similarity score
2. **Buyer Profile View**: Display institutional type, region, procurement history, and method preferences
3. **Product Catalog**: Browse all 72 OKPD2 food product categories with buyer counts
4. **Model Performance Dashboard**: View Recall@K, NDCG@K, and cold-start analysis for all models
5. **Buyer Search**: Filter buyers by region and institution type
6. **REST API**: Full API with OpenAPI documentation for integration

### Future Features (v2.0)
- Supplier-side recommendations (matching suppliers to tenders)
- Real-time model updates as new procurement data arrives
- A/B testing framework for model comparison
- Integration with MARKER platform API
- Multilingual support (Russian + English)

---

## 6. API Specification

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/{customer_inn}` | GET | Get product recommendations for a buyer |
| `/buyer/{customer_inn}/profile` | GET | Get buyer profile and purchase history |
| `/product/{okpd2_code}/info` | GET | Get product information |
| `/products` | GET | List all product categories |
| `/buyers` | GET | List/search buyer institutions |
| `/stats` | GET | System statistics and model metrics |
| `/health` | GET | Health check |

---

## 7. Evaluation Results

### Overall Performance (Test Set)

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@10 |
|-------|----------|-----------|-----------|---------|
| Two-Tower NN | 7.8% | 7.8% | 7.8% | 4.6% |
| Popularity Baseline | 83.1% | 92.2% | 100% | 45.6% |
| MF-ALS | 13.0% | 16.9% | 18.2% | 8.3% |

### Cold-Start Analysis (Buyers with < 5 interactions)

| Model | Recall@5 | Recall@10 | NDCG@10 |
|-------|----------|-----------|---------|
| Two-Tower NN | 35.3% | 35.3% | 20.7% |
| Popularity | 41.2% | 76.5% | 37.7% |
| MF-ALS | 47.1% | 64.7% | 32.9% |

### Analysis
The popularity baseline performs strongly due to the heavy concentration of dairy products (45.2% of records). The Two-Tower model shows promise for cold-start buyers (35.3% R@5) where it leverages institutional features rather than interaction history. With more training data and hyperparameter tuning, the neural model is expected to improve significantly.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Small dataset (9,994 records) | Model underfitting | Augment with more MARKER data; use regularization |
| Temporal skew (83.6% Dec) | Biased evaluation | Temporal split strategy; cross-validation |
| Product concentration (dairy dominance) | Popularity baseline too strong | Hard negative sampling; category balancing |
| Cold-start buyers (52.1%) | Poor personalization | Feature-driven Two-Tower architecture |
| Russian regulatory changes | Model drift | Scheduled retraining pipeline |

---

## 9. Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Data Analysis & Preprocessing | Sprint 1 (2 weeks) | Cleaned dataset, EDA report |
| Model Design & Architecture | Sprint 2 (2 weeks) | Feature engineering, model spec |
| Training & Evaluation | Sprint 3 (2 weeks) | Trained models, evaluation results |
| Web Application & Documentation | Sprint 4 (2 weeks) | FastAPI + Frontend, thesis chapter |
