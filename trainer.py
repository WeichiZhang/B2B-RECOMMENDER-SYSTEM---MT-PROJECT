"""
Training and Evaluation Pipeline for the Two-Tower Recommender System.

Handles:
1. Model training with early stopping
2. Evaluation (Recall@K, Precision@K, NDCG@K)
3. Cold-start stratified evaluation
4. Baseline model training and comparison
5. FAISS index building for fast retrieval
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.two_tower import (
    BuyerTower, ProductTower, TwoTowerModel,
    ProcurementDataset, collate_fn
)
from models.baselines import PopularityBaseline, MatrixFactorizationALS


# ── Evaluation Metrics ─────────────────────────────────────────────────────

def recall_at_k(ranked_list, relevant_items, k):
    """Proportion of relevant items in top-K."""
    top_k = set(ranked_list[:k])
    relevant = set(relevant_items)
    if len(relevant) == 0:
        return 0.0
    return len(top_k & relevant) / len(relevant)


def precision_at_k(ranked_list, relevant_items, k):
    """Fraction of top-K items that are relevant."""
    top_k = ranked_list[:k]
    relevant = set(relevant_items)
    return sum(1 for item in top_k if item in relevant) / k


def ndcg_at_k(ranked_list, relevant_items, k):
    """Normalized Discounted Cumulative Gain at K."""
    relevant = set(relevant_items)
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_model(score_fn, test_interactions, train_interactions, n_products,
                   ks=(5, 10, 20), n_neg=99):
    """
    Evaluate a model using the leave-one-out protocol.
    score_fn(buyer_id) -> np.array of scores for all products
    """
    # Build train positives per buyer
    train_positives = defaultdict(set)
    for bid, pid, _ in train_interactions:
        train_positives[bid].add(pid)

    # Build test positives per buyer
    test_positives = defaultdict(set)
    for bid, pid, _ in test_interactions:
        test_positives[bid].add(pid)

    # Only evaluate buyers present in both train and test
    eval_buyers = [bid for bid in test_positives if bid in train_positives]

    if len(eval_buyers) == 0:
        return {f'recall@{k}': 0.0 for k in ks}, {}

    np.random.seed(42)
    per_buyer_metrics = {}

    for bid in eval_buyers:
        # Get all test positive items for this buyer
        positives = list(test_positives[bid])

        # For each positive, evaluate against 99 random negatives
        all_items = set(range(n_products))
        known = train_positives[bid] | test_positives[bid]
        candidates = list(all_items - known)

        if len(candidates) < n_neg:
            neg_items = candidates
        else:
            neg_items = list(np.random.choice(candidates, n_neg, replace=False))

        # Get scores for candidate set
        scores = score_fn(bid)
        for pos_item in positives:
            eval_items = [pos_item] + neg_items
            item_scores = [(item, scores[item]) for item in eval_items]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            ranked = [item for item, _ in item_scores]

            metrics = {}
            for k in ks:
                metrics[f'recall@{k}'] = recall_at_k(ranked, [pos_item], k)
                metrics[f'precision@{k}'] = precision_at_k(ranked, [pos_item], k)
                metrics[f'ndcg@{k}'] = ndcg_at_k(ranked, [pos_item], k)

            per_buyer_metrics[bid] = metrics

    # Aggregate metrics
    agg = {}
    for k in ks:
        for metric_name in [f'recall@{k}', f'precision@{k}', f'ndcg@{k}']:
            values = [m[metric_name] for m in per_buyer_metrics.values()]
            agg[metric_name] = np.mean(values) if values else 0.0

    return agg, per_buyer_metrics


def cold_start_evaluation(per_buyer_metrics, train_interactions, ks=(5, 10, 20)):
    """Stratify evaluation by buyer interaction count (cold/warm/hot)."""
    # Count training interactions per buyer
    buyer_counts = defaultdict(int)
    for bid, _, _ in train_interactions:
        buyer_counts[bid] += 1

    strata = {'cold': {}, 'warm': {}, 'hot': {}}

    for bid, metrics in per_buyer_metrics.items():
        count = buyer_counts.get(bid, 0)
        if count < 5:
            strata['cold'][bid] = metrics
        elif count <= 20:
            strata['warm'][bid] = metrics
        else:
            strata['hot'][bid] = metrics

    results = {}
    for stratum_name, stratum_metrics in strata.items():
        if not stratum_metrics:
            results[stratum_name] = {'count': 0}
            continue

        agg = {'count': len(stratum_metrics)}
        for k in ks:
            for metric_name in [f'recall@{k}', f'ndcg@{k}']:
                values = [m[metric_name] for m in stratum_metrics.values()]
                agg[metric_name] = np.mean(values) if values else 0.0
        results[stratum_name] = agg

    return results


# ── Training Loop ──────────────────────────────────────────────────────────

def train_two_tower(artifacts, save_dir, epochs=100, batch_size=128,
                    lr=1e-3, weight_decay=1e-4, patience=10):
    """Train the Two-Tower model with early stopping on validation Recall@10."""
    fe = artifacts['fe']
    buyer_features = artifacts['buyer_features']
    product_features = artifacts['product_features']
    train_interactions = artifacts['train_interactions']
    val_interactions = artifacts['val_interactions']

    os.makedirs(save_dir, exist_ok=True)

    # Build dataset and dataloader
    dataset = ProcurementDataset(
        train_interactions, buyer_features, product_features, fe.n_products
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=len(dataset) > batch_size
    )

    # Initialize model
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

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Two-Tower model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_recall = 0.0
    patience_counter = 0
    training_history = []

    print(f"\nTraining Two-Tower model for up to {epochs} epochs...")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  Train interactions: {len(train_interactions)}")
    print(f"  Val interactions: {len(val_interactions)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for buyer_batch, product_batch in dataloader:
            optimizer.zero_grad()
            buyer_emb, product_emb = model(buyer_batch, product_batch)
            loss = model.compute_loss(buyer_emb, product_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()

            # Precompute all product embeddings
            product_embs = precompute_product_embeddings(model, product_features)

            def score_fn(bid):
                if bid not in buyer_features:
                    return np.zeros(fe.n_products)
                bf = buyer_features[bid]
                with torch.no_grad():
                    buyer_batch = {
                        'inst_type_ids': torch.tensor([bf['inst_type_id']], dtype=torch.long),
                        'region_ids': torch.tensor([bf['region_id']], dtype=torch.long),
                        'method_profiles': torch.tensor([bf['method_profile']]),
                        'category_profiles': torch.tensor([bf['category_profile']]),
                        'stats': torch.tensor([bf['stats']]),
                    }
                    buyer_emb = model.get_buyer_embedding(buyer_batch)
                    scores = torch.mm(buyer_emb, product_embs.t()).squeeze().numpy()
                return scores

            val_metrics, _ = evaluate_model(
                score_fn, val_interactions, train_interactions, fe.n_products
            )
            val_recall = val_metrics.get('recall@10', 0.0)

            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Val R@10: {val_recall:.4f} | R@5: {val_metrics.get('recall@5', 0):.4f} | "
                  f"NDCG@10: {val_metrics.get('ndcg@10', 0):.4f}")

            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'val_metrics': val_metrics,
            })

            # Early stopping
            if val_recall > best_recall:
                best_recall = val_recall
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience // 5 + 1:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

    # Load best model
    best_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2, default=str)

    return model, training_history


def precompute_product_embeddings(model, product_features):
    """Compute embeddings for all products."""
    model.eval()
    embs = []
    with torch.no_grad():
        for pid in sorted(product_features.keys()):
            pf = product_features[pid]
            product_batch = {
                'okpd2_group_ids': torch.tensor([pf['okpd2_group_id']], dtype=torch.long),
                'okpd2_full_ids': torch.tensor([pf['okpd2_full_id']], dtype=torch.long),
                'ktru_ids': torch.tensor([pf['ktru_id']], dtype=torch.long),
                'unit_ids': torch.tensor([pf['unit_id']], dtype=torch.long),
                'tfidf_vecs': torch.tensor([pf['tfidf_vec']]),
                'prices': torch.tensor([pf['scaled_price']]),
            }
            emb = model.get_product_embedding(product_batch)
            embs.append(emb)
    return torch.cat(embs, dim=0)


def build_faiss_index(product_embeddings):
    """Build a FAISS index from product embeddings for fast retrieval."""
    import faiss
    dim = product_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim on L2-normalized vectors)
    index.add(product_embeddings.numpy().astype(np.float32))
    return index


# ── Full Pipeline ──────────────────────────────────────────────────────────

def run_full_pipeline(data_path, save_dir='./saved_models'):
    """Run the complete training and evaluation pipeline."""
    from data.preprocessing import run_pipeline

    print("=" * 70)
    print("B2B PROCUREMENT RECOMMENDER SYSTEM")
    print("Two-Tower Neural Network Training Pipeline")
    print("=" * 70)

    # Step 1: Preprocess data
    artifacts = run_pipeline(data_path)
    fe = artifacts['fe']

    # Save feature engineer
    os.makedirs(save_dir, exist_ok=True)
    fe.save(os.path.join(save_dir, 'feature_engineer.pkl'))

    # Step 2: Train Two-Tower model
    print("\n" + "=" * 70)
    print("TRAINING TWO-TOWER MODEL")
    print("=" * 70)
    model, history = train_two_tower(artifacts, save_dir)

    # Step 3: Train baselines
    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)

    # Popularity baseline
    pop_baseline = PopularityBaseline()
    pop_baseline.fit(artifacts['train_interactions'], fe.n_products)
    print("Popularity baseline fitted.")

    # Matrix Factorization
    mf_baseline = MatrixFactorizationALS(n_factors=64, n_iterations=15)
    mf_baseline.fit(artifacts['train_interactions'], fe.n_buyers, fe.n_products)

    # Step 4: Evaluate all models on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    # Precompute Two-Tower product embeddings
    product_embs = precompute_product_embeddings(model, artifacts['product_features'])

    # Build FAISS index
    faiss_index = build_faiss_index(product_embs)
    import faiss
    faiss.write_index(faiss_index, os.path.join(save_dir, 'product_index.faiss'))

    # Save product embeddings
    np.save(os.path.join(save_dir, 'product_embeddings.npy'),
            product_embs.numpy())

    # Two-Tower scoring function
    def tt_score_fn(bid):
        if bid not in artifacts['buyer_features']:
            return np.zeros(fe.n_products)
        bf = artifacts['buyer_features'][bid]
        with torch.no_grad():
            buyer_batch = {
                'inst_type_ids': torch.tensor([bf['inst_type_id']], dtype=torch.long),
                'region_ids': torch.tensor([bf['region_id']], dtype=torch.long),
                'method_profiles': torch.tensor([bf['method_profile']]),
                'category_profiles': torch.tensor([bf['category_profile']]),
                'stats': torch.tensor([bf['stats']]),
            }
            buyer_emb = model.get_buyer_embedding(buyer_batch)
            scores = torch.mm(buyer_emb, product_embs.t()).squeeze().numpy()
        return scores

    results = {}

    # Evaluate Two-Tower
    print("\nEvaluating Two-Tower model...")
    tt_metrics, tt_per_buyer = evaluate_model(
        tt_score_fn, artifacts['test_interactions'],
        artifacts['train_interactions'], fe.n_products
    )
    results['two_tower'] = tt_metrics
    print(f"  Two-Tower: {tt_metrics}")

    # Evaluate Popularity
    print("Evaluating Popularity baseline...")
    pop_metrics, pop_per_buyer = evaluate_model(
        lambda bid: pop_baseline.get_scores(bid),
        artifacts['test_interactions'],
        artifacts['train_interactions'], fe.n_products
    )
    results['popularity'] = pop_metrics
    print(f"  Popularity: {pop_metrics}")

    # Evaluate MF-ALS
    print("Evaluating MF-ALS baseline...")
    mf_metrics, mf_per_buyer = evaluate_model(
        lambda bid: mf_baseline.get_scores(bid),
        artifacts['test_interactions'],
        artifacts['train_interactions'], fe.n_products
    )
    results['mf_als'] = mf_metrics
    print(f"  MF-ALS: {mf_metrics}")

    # Step 5: Cold-start evaluation
    print("\n" + "=" * 70)
    print("COLD-START EVALUATION")
    print("=" * 70)

    cold_results = {}
    for name, per_buyer in [('two_tower', tt_per_buyer),
                             ('popularity', pop_per_buyer),
                             ('mf_als', mf_per_buyer)]:
        cs = cold_start_evaluation(per_buyer, artifacts['train_interactions'])
        cold_results[name] = cs
        print(f"\n{name}:")
        for stratum, metrics in cs.items():
            print(f"  {stratum}: {metrics}")

    # Save all results
    all_results = {
        'overall': results,
        'cold_start': cold_results,
        'model_params': sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'NDCG@10':>8}")
    print("-" * 56)
    for name in ['two_tower', 'popularity', 'mf_als']:
        m = results[name]
        print(f"{name:<20} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} "
              f"{m.get('recall@20',0):>8.4f} {m.get('ndcg@10',0):>8.4f}")

    # Save artifacts needed for the webapp
    import pickle
    webapp_artifacts = {
        'fe': fe,
        'buyer_features': artifacts['buyer_features'],
        'product_features': artifacts['product_features'],
        'train_interactions': artifacts['train_interactions'],
        'results': all_results,
    }
    with open(os.path.join(save_dir, 'webapp_artifacts.pkl'), 'wb') as f:
        pickle.dump(webapp_artifacts, f)

    print(f"\nAll artifacts saved to: {save_dir}")
    return model, all_results


if __name__ == '__main__':
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'b2b_procurement_merged_dataset.xlsx')
    run_full_pipeline(data_path, save_dir=os.path.join(
        os.path.dirname(__file__), '..', 'saved_models'))
