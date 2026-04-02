"""
Two-Tower Neural Network Model for B2B Procurement Recommendations.

Architecture:
- Buyer Tower: encodes institutional features (type, region, method profile,
  category history, contract stats) into a 64-dim embedding
- Product Tower: encodes product features (OKPD2 hierarchy, KTRU, TF-IDF text,
  unit, price) into a 64-dim embedding
- Similarity: cosine similarity between buyer and product embeddings
- Loss: sampled softmax with in-batch negatives + hard negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BuyerTower(nn.Module):
    """Encodes buyer institutional features into a dense embedding."""

    def __init__(self, n_inst_types, n_regions, n_products, inst_emb_dim=8,
                 region_emb_dim=16, hidden_dims=(256, 128), output_dim=64,
                 dropout=0.2):
        super().__init__()
        self.inst_embedding = nn.Embedding(n_inst_types + 1, inst_emb_dim)
        self.region_embedding = nn.Embedding(n_regions + 1, region_emb_dim)

        # Input: inst_emb + region_emb + method_profile(4) + category_profile(n_products) + stats(2)
        input_dim = inst_emb_dim + region_emb_dim + 4 + n_products + 2

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inst_type_ids, region_ids, method_profiles,
                category_profiles, stats):
        inst_emb = self.inst_embedding(inst_type_ids)
        region_emb = self.region_embedding(region_ids)
        x = torch.cat([inst_emb, region_emb, method_profiles,
                        category_profiles, stats], dim=1)
        x = self.network(x)
        return F.normalize(x, p=2, dim=1)


class ProductTower(nn.Module):
    """Encodes product features into a dense embedding."""

    def __init__(self, n_okpd2_groups, n_okpd2_full, n_ktru, n_units,
                 tfidf_dim=500, group_emb_dim=16, full_emb_dim=16,
                 ktru_emb_dim=32, unit_emb_dim=4, text_proj_dim=64,
                 hidden_dims=(256, 128), output_dim=64, dropout=0.2):
        super().__init__()
        self.okpd2_group_emb = nn.Embedding(n_okpd2_groups + 1, group_emb_dim)
        self.okpd2_full_emb = nn.Embedding(n_okpd2_full + 1, full_emb_dim)
        self.ktru_emb = nn.Embedding(n_ktru + 1, ktru_emb_dim)
        self.unit_emb = nn.Embedding(n_units + 1, unit_emb_dim)

        # Text projection
        self.text_proj = nn.Linear(tfidf_dim, text_proj_dim)

        # Input: group_emb + full_emb + ktru_emb + unit_emb + text_proj + price(1)
        input_dim = group_emb_dim + full_emb_dim + ktru_emb_dim + unit_emb_dim + text_proj_dim + 1

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, okpd2_group_ids, okpd2_full_ids, ktru_ids, unit_ids,
                tfidf_vecs, prices):
        group_emb = self.okpd2_group_emb(okpd2_group_ids)
        full_emb = self.okpd2_full_emb(okpd2_full_ids)
        ktru_emb = self.ktru_emb(ktru_ids)
        unit_emb = self.unit_emb(unit_ids)
        text_emb = F.relu(self.text_proj(tfidf_vecs))

        x = torch.cat([group_emb, full_emb, ktru_emb, unit_emb,
                        text_emb, prices.unsqueeze(1)], dim=1)
        x = self.network(x)
        return F.normalize(x, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Complete Two-Tower model combining buyer and product towers."""

    def __init__(self, buyer_tower, product_tower, temperature=0.07):
        super().__init__()
        self.buyer_tower = buyer_tower
        self.product_tower = product_tower
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, buyer_batch, product_batch):
        buyer_emb = self.buyer_tower(**buyer_batch)
        product_emb = self.product_tower(**product_batch)
        return buyer_emb, product_emb

    def compute_loss(self, buyer_emb, product_emb):
        """In-batch sampled softmax loss."""
        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.mm(buyer_emb, product_emb.t()) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def get_buyer_embedding(self, buyer_batch):
        return self.buyer_tower(**buyer_batch)

    def get_product_embedding(self, product_batch):
        return self.product_tower(**product_batch)


class ProcurementDataset(torch.utils.data.Dataset):
    """Dataset for buyer-product interaction pairs."""

    def __init__(self, interactions, buyer_features, product_features, n_products,
                 neg_ratio=4):
        self.interactions = interactions
        self.buyer_features = buyer_features
        self.product_features = product_features
        self.n_products = n_products
        self.neg_ratio = neg_ratio

        # Build positive set per buyer for negative sampling
        self.buyer_positives = {}
        for bid, pid, _ in interactions:
            if bid not in self.buyer_positives:
                self.buyer_positives[bid] = set()
            self.buyer_positives[bid].add(pid)

        # Product popularity for hard negative sampling
        product_counts = {}
        for _, pid, _ in interactions:
            product_counts[pid] = product_counts.get(pid, 0) + 1
        total = sum(product_counts.values())
        self.product_probs = np.zeros(n_products)
        for pid, count in product_counts.items():
            self.product_probs[pid] = count / total

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        bid, pid, _ = self.interactions[idx]

        buyer_feat = self.buyer_features.get(bid)
        product_feat = self.product_features.get(pid)

        if buyer_feat is None or product_feat is None:
            # Fallback to first valid interaction
            for b, p, _ in self.interactions:
                if b in self.buyer_features and p in self.product_features:
                    buyer_feat = self.buyer_features[b]
                    product_feat = self.product_features[p]
                    break

        return {
            'buyer': {
                'inst_type_ids': torch.tensor(buyer_feat['inst_type_id'], dtype=torch.long),
                'region_ids': torch.tensor(buyer_feat['region_id'], dtype=torch.long),
                'method_profiles': torch.tensor(buyer_feat['method_profile'], dtype=torch.float32),
                'category_profiles': torch.tensor(buyer_feat['category_profile'], dtype=torch.float32),
                'stats': torch.tensor(buyer_feat['stats'], dtype=torch.float32),
            },
            'product': {
                'okpd2_group_ids': torch.tensor(product_feat['okpd2_group_id'], dtype=torch.long),
                'okpd2_full_ids': torch.tensor(product_feat['okpd2_full_id'], dtype=torch.long),
                'ktru_ids': torch.tensor(product_feat['ktru_id'], dtype=torch.long),
                'unit_ids': torch.tensor(product_feat['unit_id'], dtype=torch.long),
                'tfidf_vecs': torch.tensor(product_feat['tfidf_vec'], dtype=torch.float32),
                'prices': torch.tensor(product_feat['scaled_price'], dtype=torch.float32),
            }
        }


def collate_fn(batch):
    """Custom collate to batch buyer and product features separately."""
    buyer_batch = {
        'inst_type_ids': torch.stack([b['buyer']['inst_type_ids'] for b in batch]),
        'region_ids': torch.stack([b['buyer']['region_ids'] for b in batch]),
        'method_profiles': torch.stack([b['buyer']['method_profiles'] for b in batch]),
        'category_profiles': torch.stack([b['buyer']['category_profiles'] for b in batch]),
        'stats': torch.stack([b['buyer']['stats'] for b in batch]),
    }
    product_batch = {
        'okpd2_group_ids': torch.stack([b['product']['okpd2_group_ids'] for b in batch]),
        'okpd2_full_ids': torch.stack([b['product']['okpd2_full_ids'] for b in batch]),
        'ktru_ids': torch.stack([b['product']['ktru_ids'] for b in batch]),
        'unit_ids': torch.stack([b['product']['unit_ids'] for b in batch]),
        'tfidf_vecs': torch.stack([b['product']['tfidf_vecs'] for b in batch]),
        'prices': torch.stack([b['product']['prices'] for b in batch]),
    }
    return buyer_batch, product_batch
