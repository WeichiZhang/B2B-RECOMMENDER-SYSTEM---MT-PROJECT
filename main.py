"""
FastAPI Backend for B2B Procurement Recommender System.

Endpoints:
- GET /recommend/{customer_inn} - Get product recommendations for a buyer
- GET /buyer/{customer_inn}/profile - Get buyer profile and purchase history
- GET /product/{okpd2_code}/info - Get product information
- GET /stats - Get system statistics and model performance
- GET /products - List all available products
- GET /buyers - List all buyers with search/filter
"""

import os
import pickle
import numpy as np
import torch
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from collections import defaultdict

# ── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="B2B Procurement Recommender API",
    description="Two-Tower Neural Network based recommendation system for "
                "B2B food procurement in the Russian market",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────────

STATE = {
    'model': None,
    'fe': None,
    'buyer_features': None,
    'product_features': None,
    'faiss_index': None,
    'product_embeddings': None,
    'train_interactions': None,
    'results': None,
    'inn_to_bid': None,
    'bid_to_inn': None,
    'pid_to_code': None,
    'code_to_pid': None,
}


# ── Response Models ────────────────────────────────────────────────────────

class ProductRecommendation(BaseModel):
    rank: int
    okpd2_code: str
    product_name: str
    description: str
    similarity_score: float


class RecommendationResponse(BaseModel):
    customer_inn: int
    institution_type: str
    region: str
    recommendations: List[ProductRecommendation]
    model_used: str = "Two-Tower Neural Network"


class BuyerProfile(BaseModel):
    customer_inn: int
    institution_type: str
    region: str
    n_purchases: int
    purchased_products: List[Dict[str, Any]]
    method_profile: Dict[str, float]


class ProductInfo(BaseModel):
    okpd2_code: str
    product_name: str
    description: str
    mean_price: float
    unit: str
    n_buyers: int


class SystemStats(BaseModel):
    n_buyers: int
    n_products: int
    n_regions: int
    n_interactions: int
    model_performance: Dict[str, Any]
    cold_start_analysis: Dict[str, Any]


# ── Model Loading ──────────────────────────────────────────────────────────

def load_model_artifacts():
    """Load all saved model artifacts at startup."""
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

    # Load webapp artifacts
    artifacts_path = os.path.join(save_dir, 'webapp_artifacts.pkl')
    if not os.path.exists(artifacts_path):
        print("WARNING: No saved model artifacts found. Run trainer.py first.")
        return False

    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    STATE['fe'] = artifacts['fe']
    STATE['buyer_features'] = artifacts['buyer_features']
    STATE['product_features'] = artifacts['product_features']
    STATE['train_interactions'] = artifacts['train_interactions']
    STATE['results'] = artifacts['results']

    # Build reverse mappings
    fe = STATE['fe']
    STATE['inn_to_bid'] = fe.buyer_id_map
    STATE['bid_to_inn'] = {v: k for k, v in fe.buyer_id_map.items()}
    STATE['code_to_pid'] = fe.product_id_map
    STATE['pid_to_code'] = {v: k for k, v in fe.product_id_map.items()}

    # Load FAISS index
    index_path = os.path.join(save_dir, 'product_index.faiss')
    if os.path.exists(index_path):
        STATE['faiss_index'] = faiss.read_index(index_path)

    # Load product embeddings
    emb_path = os.path.join(save_dir, 'product_embeddings.npy')
    if os.path.exists(emb_path):
        STATE['product_embeddings'] = np.load(emb_path)

    # Load PyTorch model
    from models.two_tower import BuyerTower, ProductTower, TwoTowerModel
    model_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(model_path):
        buyer_tower = BuyerTower(
            n_inst_types=fe.n_inst_types,
            n_regions=fe.n_regions,
            n_products=fe.n_products,
        )
        product_tower = ProductTower(
            n_okpd2_groups=fe.n_okpd2_groups,
            n_okpd2_full=fe.n_okpd2_full,
            n_ktru=fe.n_ktru,
            n_units=fe.n_units,
            tfidf_dim=fe.tfidf_dim,
        )
        model = TwoTowerModel(buyer_tower, product_tower)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        STATE['model'] = model

    print("Model artifacts loaded successfully!")
    return True


@app.on_event("startup")
async def startup_event():
    load_model_artifacts()


# ── Helper Functions ───────────────────────────────────────────────────────

def get_buyer_embedding(buyer_id):
    """Compute buyer embedding using the Two-Tower model."""
    model = STATE['model']
    bf = STATE['buyer_features'].get(buyer_id)
    if model is None or bf is None:
        return None

    with torch.no_grad():
        buyer_batch = {
            'inst_type_ids': torch.tensor([bf['inst_type_id']], dtype=torch.long),
            'region_ids': torch.tensor([bf['region_id']], dtype=torch.long),
            'method_profiles': torch.tensor([bf['method_profile']]),
            'category_profiles': torch.tensor([bf['category_profile']]),
            'stats': torch.tensor([bf['stats']]),
        }
        emb = model.get_buyer_embedding(buyer_batch)
    return emb.numpy()


def get_recommendations_for_buyer(buyer_id, k=10, exclude_purchased=True):
    """Get top-K product recommendations for a buyer."""
    buyer_emb = get_buyer_embedding(buyer_id)
    if buyer_emb is None:
        return []

    product_embs = STATE['product_embeddings']
    if product_embs is None:
        return []

    # Compute cosine similarities
    scores = (buyer_emb @ product_embs.T).flatten()

    # Optionally exclude already-purchased products
    if exclude_purchased:
        purchased = set()
        for bid, pid, _ in STATE['train_interactions']:
            if bid == buyer_id:
                purchased.add(pid)
        for pid in purchased:
            if pid < len(scores):
                scores[pid] = -np.inf

    # Get top-K
    top_indices = np.argsort(scores)[::-1][:k]
    recommendations = []
    for rank, pid in enumerate(top_indices):
        if scores[pid] == -np.inf:
            continue
        pf = STATE['product_features'].get(pid, {})
        recommendations.append({
            'rank': rank + 1,
            'okpd2_code': pf.get('okpd2_code', STATE['pid_to_code'].get(pid, 'unknown')),
            'product_name': pf.get('product_name', 'Unknown'),
            'description': pf.get('description', ''),
            'similarity_score': float(scores[pid]),
        })

    return recommendations


# ── API Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "B2B Procurement Recommender System",
        "version": "1.0.0",
        "description": "Two-Tower Neural Network for food procurement recommendations",
        "endpoints": [
            "/recommend/{customer_inn}",
            "/buyer/{customer_inn}/profile",
            "/product/{okpd2_code}/info",
            "/products",
            "/buyers",
            "/stats",
        ]
    }


@app.get("/recommend/{customer_inn}", response_model=RecommendationResponse)
async def get_recommendations(
    customer_inn: int,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    exclude_purchased: bool = Query(True, description="Exclude already-purchased products"),
):
    """Get product recommendations for a buyer institution."""
    if STATE['model'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    bid = STATE['inn_to_bid'].get(customer_inn)
    if bid is None:
        raise HTTPException(status_code=404, detail=f"Buyer INN {customer_inn} not found")

    bf = STATE['buyer_features'].get(bid)
    if bf is None:
        raise HTTPException(status_code=404, detail="Buyer features not available")

    recommendations = get_recommendations_for_buyer(bid, k, exclude_purchased)

    # Get buyer info
    fe = STATE['fe']
    inst_type_map = {v: k for k, v in fe.inst_type_id_map.items()}
    region_map = {v: k for k, v in fe.region_id_map.items()}

    return RecommendationResponse(
        customer_inn=customer_inn,
        institution_type=inst_type_map.get(bf['inst_type_id'], 'unknown'),
        region=region_map.get(bf['region_id'], 'unknown'),
        recommendations=[ProductRecommendation(**r) for r in recommendations],
    )


@app.get("/buyer/{customer_inn}/profile", response_model=BuyerProfile)
async def get_buyer_profile(customer_inn: int):
    """Get buyer institution profile and purchase history."""
    bid = STATE['inn_to_bid'].get(customer_inn)
    if bid is None:
        raise HTTPException(status_code=404, detail=f"Buyer INN {customer_inn} not found")

    bf = STATE['buyer_features'].get(bid)
    if bf is None:
        raise HTTPException(status_code=404, detail="Buyer features not available")

    fe = STATE['fe']
    inst_type_map = {v: k for k, v in fe.inst_type_id_map.items()}
    region_map = {v: k for k, v in fe.region_id_map.items()}

    # Get purchased products
    purchased = []
    for b, p, ts in STATE['train_interactions']:
        if b == bid:
            pf = STATE['product_features'].get(p, {})
            purchased.append({
                'okpd2_code': pf.get('okpd2_code', ''),
                'product_name': pf.get('product_name', ''),
                'description': pf.get('description', ''),
            })

    # Method profile
    tender_types = ['Electronic auction', 'Open tender',
                    'Electronic quotation request', 'Single supplier purchase']
    method_dict = {}
    for i, tt in enumerate(tender_types):
        if i < len(bf['method_profile']):
            method_dict[tt] = float(bf['method_profile'][i])

    return BuyerProfile(
        customer_inn=customer_inn,
        institution_type=inst_type_map.get(bf['inst_type_id'], 'unknown'),
        region=region_map.get(bf['region_id'], 'unknown'),
        n_purchases=len(purchased),
        purchased_products=purchased,
        method_profile=method_dict,
    )


@app.get("/product/{okpd2_code}/info", response_model=ProductInfo)
async def get_product_info(okpd2_code: str):
    """Get detailed product information."""
    pid = STATE['code_to_pid'].get(okpd2_code)
    if pid is None:
        raise HTTPException(status_code=404, detail=f"Product {okpd2_code} not found")

    pf = STATE['product_features'].get(pid)
    if pf is None:
        raise HTTPException(status_code=404, detail="Product features not available")

    # Count unique buyers
    buyer_count = len(set(b for b, p, _ in STATE['train_interactions'] if p == pid))

    fe = STATE['fe']
    unit_map = {v: k for k, v in fe.unit_map.items()}

    return ProductInfo(
        okpd2_code=okpd2_code,
        product_name=pf.get('product_name', 'Unknown'),
        description=pf.get('description', ''),
        mean_price=float(pf.get('scaled_price', 0)),
        unit=unit_map.get(pf.get('unit_id', 0), 'kg'),
        n_buyers=buyer_count,
    )


@app.get("/products")
async def list_products():
    """List all available food product categories."""
    products = []
    for pid, pf in sorted(STATE['product_features'].items()):
        buyer_count = len(set(b for b, p, _ in STATE['train_interactions'] if p == pid))
        products.append({
            'product_id': pid,
            'okpd2_code': pf.get('okpd2_code', ''),
            'product_name': pf.get('product_name', ''),
            'description': pf.get('description', ''),
            'n_buyers': buyer_count,
        })
    return {"products": products, "total": len(products)}


@app.get("/buyers")
async def list_buyers(
    region: Optional[str] = Query(None, description="Filter by region"),
    inst_type: Optional[str] = Query(None, description="Filter by institution type"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all buyer institutions with optional filtering."""
    fe = STATE['fe']
    inst_type_map = {v: k for k, v in fe.inst_type_id_map.items()}
    region_map = {v: k for k, v in fe.region_id_map.items()}

    buyers = []
    for bid, bf in STATE['buyer_features'].items():
        buyer_inst = inst_type_map.get(bf['inst_type_id'], 'unknown')
        buyer_region = region_map.get(bf['region_id'], 'unknown')

        if region and region.lower() not in buyer_region.lower():
            continue
        if inst_type and inst_type.lower() not in buyer_inst.lower():
            continue

        n_purchases = sum(1 for b, _, _ in STATE['train_interactions'] if b == bid)
        buyers.append({
            'buyer_id': bid,
            'customer_inn': bf['inn'],
            'institution_type': buyer_inst,
            'region': buyer_region,
            'n_purchases': n_purchases,
        })

    total = len(buyers)
    buyers = buyers[offset:offset + limit]
    return {"buyers": buyers, "total": total, "limit": limit, "offset": offset}


@app.get("/stats", response_model=SystemStats)
async def get_stats():
    """Get system statistics and model performance metrics."""
    fe = STATE['fe']
    results = STATE.get('results', {})

    return SystemStats(
        n_buyers=fe.n_buyers,
        n_products=fe.n_products,
        n_regions=fe.n_regions,
        n_interactions=len(STATE.get('train_interactions', [])),
        model_performance=results.get('overall', {}),
        cold_start_analysis=results.get('cold_start', {}),
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": STATE['model'] is not None,
        "index_loaded": STATE['faiss_index'] is not None,
    }
