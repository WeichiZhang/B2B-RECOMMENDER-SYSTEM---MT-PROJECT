"""
Data Preprocessing and Feature Engineering Pipeline
for B2B Procurement Recommender System.

This module handles:
1. Loading and cleaning the merged procurement dataset
2. Feature engineering for Buyer Tower and Product Tower
3. Interaction matrix construction
4. Train/Validation/Test temporal splitting
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import csr_matrix


# ── Institutional Type Extraction ──────────────────────────────────────────
INSTITUTION_TYPES = {
    'МБДОУ': 'preschool',
    'МАДОУ': 'preschool',
    'МКДОУ': 'preschool',
    'МБОУ': 'school',
    'МАОУ': 'school',
    'МКОУ': 'school',
    'ГБОУ': 'state_school',
    'ГБУЗ': 'hospital',
    'ГАУЗ': 'hospital',
    'ГКУЗ': 'hospital',
    'БУ': 'budget_institution',
    'ФГБУ': 'federal_institution',
    'ФГБОУ': 'federal_education',
    'МБУ': 'municipal_institution',
    'МКУ': 'municipal_institution',
    'КУ': 'state_institution',
    'АУ': 'autonomous_institution',
}


def extract_institution_type(name):
    """Extract institutional type from Russian organization name prefix."""
    if not isinstance(name, str):
        return 'other'
    name_upper = name.strip().upper()
    # Try longest prefix first
    for prefix in sorted(INSTITUTION_TYPES.keys(), key=len, reverse=True):
        pattern = r'^["\s«]*' + re.escape(prefix) + r'[\s"]'
        if re.match(pattern, name_upper):
            return INSTITUTION_TYPES[prefix]
    return 'other'


def load_and_clean(filepath):
    """Load the merged dataset and apply cleaning operations."""
    df = pd.read_excel(filepath)

    # Remove the entirely-null electronic_trading column
    df = df.drop(columns=['electronic_trading'], errors='ignore')

    # Remove unclassified tender type
    df = df[df['tender_type'] != '-'].copy()

    # Deduplicate on (customer_inn, registry_number, okpd2_code)
    df = df.drop_duplicates(subset=['customer_inn', 'registry_number', 'okpd2_code'])

    # Parse dates
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

    # Fill missing values
    df['ktru_code_clean'] = df['ktru_code_clean'].fillna('UNKNOWN')
    df['ktru_description'] = df['ktru_description'].fillna('UNKNOWN')
    df['vat_rate_pct'] = df['vat_rate_pct'].fillna(10.0)

    # Derive missing quantity from cost / price where possible
    mask = df['quantity'].isna() & (df['price_incl_vat_rub'] > 0)
    df.loc[mask, 'quantity'] = df.loc[mask, 'total_cost_incl_vat_rub'] / df.loc[mask, 'price_incl_vat_rub']
    df['quantity'] = df['quantity'].fillna(0)

    # Derive missing total_cost
    mask = df['total_cost_incl_vat_rub'].isna()
    df.loc[mask, 'total_cost_incl_vat_rub'] = df.loc[mask, 'quantity'] * df.loc[mask, 'price_incl_vat_rub']

    # Extract institution type
    df['institution_type'] = df['customer_name'].apply(extract_institution_type)

    # Log-transform numeric features
    for col in ['price_incl_vat_rub', 'customer_cost_rub', 'supplier_cost_rub', 'quantity']:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    # Extract OKPD2 hierarchy levels
    df['okpd2_group'] = df['okpd2_code'].str[:5]  # e.g., "10.51"
    df['okpd2_class'] = df['okpd2_code'].str[:2]   # e.g., "10"

    return df


def temporal_split(df):
    """
    Split data temporally. Since 83.6% of data is in Dec 2025,
    we use: train (<=Nov 2025), val (Dec 1-15), test (Dec 16+).
    If pre-December data is too small, we split December into 60/20/20.
    """
    df = df.copy()
    pre_dec = df[df['publication_date'] < '2025-12-01']
    dec_onward = df[df['publication_date'] >= '2025-12-01']

    if len(pre_dec) >= 500:
        # Enough pre-December data
        train = pre_dec
        dec_sorted = dec_onward.sort_values('publication_date')
        mid = int(len(dec_sorted) * 0.5)
        val = dec_sorted.iloc[:mid]
        test = dec_sorted.iloc[mid:]
    else:
        # Not enough pre-December data, split all data 60/20/20
        df_sorted = df.sort_values('publication_date').reset_index(drop=True)
        n = len(df_sorted)
        train = df_sorted.iloc[:int(n * 0.6)]
        val = df_sorted.iloc[int(n * 0.6):int(n * 0.8)]
        test = df_sorted.iloc[int(n * 0.8):]

    return train, val, test


class FeatureEngineer:
    """Builds buyer and product feature vectors from raw procurement data."""

    def __init__(self):
        self.buyer_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        self.inst_type_encoder = LabelEncoder()
        self.tender_types = ['Electronic auction', 'Open tender',
                             'Electronic quotation request',
                             'Single supplier purchase (Art.93 pt.12 44-FZ)']
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english',
                                     lowercase=True)
        self.price_scaler = StandardScaler()
        self.buyer_stats_scaler = StandardScaler()

        # Learned mappings
        self.buyer_id_map = {}
        self.product_id_map = {}
        self.region_id_map = {}
        self.inst_type_id_map = {}
        self.okpd2_group_map = {}
        self.okpd2_full_map = {}
        self.ktru_map = {}
        self.unit_map = {}

        # Feature dimensions
        self.n_buyers = 0
        self.n_products = 0
        self.n_regions = 0
        self.n_inst_types = 0
        self.n_okpd2_groups = 0
        self.n_okpd2_full = 0
        self.n_ktru = 0
        self.n_units = 0
        self.tfidf_dim = 1000

    def fit(self, df):
        """Fit all encoders and scalers on training data."""
        # Encode buyer and product IDs
        buyers = df['customer_inn'].unique()
        products = df['okpd2_code'].unique()

        self.buyer_id_map = {inn: i for i, inn in enumerate(buyers)}
        self.product_id_map = {code: i for i, code in enumerate(products)}
        self.n_buyers = len(buyers)
        self.n_products = len(products)

        # Region encoding
        regions = df['delivery_region'].unique()
        self.region_id_map = {r: i for i, r in enumerate(regions)}
        self.n_regions = len(regions)

        # Institution type encoding
        inst_types = df['institution_type'].unique()
        self.inst_type_id_map = {t: i for i, t in enumerate(inst_types)}
        self.n_inst_types = len(inst_types)

        # OKPD2 group encoding
        groups = df['okpd2_group'].unique()
        self.okpd2_group_map = {g: i for i, g in enumerate(groups)}
        self.n_okpd2_groups = len(groups)

        # OKPD2 full code encoding
        full_codes = df['okpd2_code'].unique()
        self.okpd2_full_map = {c: i for i, c in enumerate(full_codes)}
        self.n_okpd2_full = len(full_codes)

        # KTRU encoding (with UNKNOWN token)
        ktru_codes = list(df['ktru_code_clean'].unique())
        if 'UNKNOWN' not in ktru_codes:
            ktru_codes.append('UNKNOWN')
        self.ktru_map = {k: i for i, k in enumerate(ktru_codes)}
        self.n_ktru = len(ktru_codes)

        # Unit encoding
        units = df['unit'].unique()
        self.unit_map = {u: i for i, u in enumerate(units)}
        self.n_units = len(units)

        # Fit TF-IDF on product names
        self.tfidf.fit(df['product_name_en'].fillna(''))
        self.tfidf_dim = len(self.tfidf.vocabulary_)

        # Fit price scaler
        self.price_scaler.fit(df[['price_incl_vat_rub_log']].values)

        # Fit buyer stats scaler
        buyer_stats = df.groupby('customer_inn').agg({
            'customer_cost_rub_log': 'mean',
            'price_incl_vat_rub_log': 'mean',
        }).values
        self.buyer_stats_scaler.fit(buyer_stats)

        return self

    def build_buyer_features(self, df):
        """Build per-buyer feature dictionaries from training data."""
        buyer_features = {}

        for inn, group in df.groupby('customer_inn'):
            if inn not in self.buyer_id_map:
                continue

            bid = self.buyer_id_map[inn]

            # Institution type (integer ID)
            inst_type = group['institution_type'].iloc[0]
            inst_type_id = self.inst_type_id_map.get(inst_type, 0)

            # Region (integer ID)
            region = group['delivery_region'].iloc[0]
            region_id = self.region_id_map.get(region, 0)

            # Procurement method profile (4-dim normalized vector)
            tender_counts = group['tender_type'].value_counts()
            method_profile = np.zeros(len(self.tender_types))
            for i, tt in enumerate(self.tender_types):
                method_profile[i] = tender_counts.get(tt, 0)
            total = method_profile.sum()
            if total > 0:
                method_profile /= total

            # Historical category profile (multi-hot with TF-IDF weighting)
            cat_counts = group['okpd2_code'].value_counts()
            category_profile = np.zeros(self.n_products)
            for code, count in cat_counts.items():
                if code in self.product_id_map:
                    category_profile[self.product_id_map[code]] = count
            total = category_profile.sum()
            if total > 0:
                category_profile /= total

            # Aggregate contract statistics
            mean_cost = group['customer_cost_rub_log'].mean()
            mean_price = group['price_incl_vat_rub_log'].mean()
            stats = self.buyer_stats_scaler.transform([[mean_cost, mean_price]])[0]

            buyer_features[bid] = {
                'buyer_id': bid,
                'inn': inn,
                'inst_type_id': inst_type_id,
                'region_id': region_id,
                'method_profile': method_profile.astype(np.float32),
                'category_profile': category_profile.astype(np.float32),
                'stats': stats.astype(np.float32),
            }

        return buyer_features

    def build_product_features(self, df):
        """Build per-product feature dictionaries."""
        product_features = {}

        for code, group in df.groupby('okpd2_code'):
            if code not in self.product_id_map:
                continue

            pid = self.product_id_map[code]

            # OKPD2 hierarchy
            okpd2_group_id = self.okpd2_group_map.get(group['okpd2_group'].iloc[0], 0)
            okpd2_full_id = self.okpd2_full_map.get(code, 0)

            # KTRU code
            ktru = group['ktru_code_clean'].mode()
            ktru_code = ktru.iloc[0] if len(ktru) > 0 else 'UNKNOWN'
            ktru_id = self.ktru_map.get(ktru_code, self.ktru_map.get('UNKNOWN', 0))

            # TF-IDF text features from product name
            product_names = group['product_name_en'].fillna('').unique()
            tfidf_vec = self.tfidf.transform([' '.join(product_names)]).toarray()[0]

            # Unit encoding
            unit = group['unit'].mode()
            unit_val = unit.iloc[0] if len(unit) > 0 else 'kg'
            unit_id = self.unit_map.get(unit_val, 0)

            # Mean log price (scaled)
            mean_price = group['price_incl_vat_rub_log'].mean()
            scaled_price = self.price_scaler.transform([[mean_price]])[0][0]

            # Product description
            desc = group['okpd2_description'].iloc[0] if 'okpd2_description' in group.columns else ''

            product_features[pid] = {
                'product_id': pid,
                'okpd2_code': code,
                'okpd2_group_id': okpd2_group_id,
                'okpd2_full_id': okpd2_full_id,
                'ktru_id': ktru_id,
                'unit_id': unit_id,
                'tfidf_vec': tfidf_vec.astype(np.float32),
                'scaled_price': np.float32(scaled_price),
                'description': desc,
                'product_name': group['product_name_en'].iloc[0],
            }

        return product_features

    def build_interactions(self, df):
        """Build interaction list: [(buyer_id, product_id, timestamp), ...]."""
        interactions = []
        for _, row in df.iterrows():
            inn = row['customer_inn']
            code = row['okpd2_code']
            if inn in self.buyer_id_map and code in self.product_id_map:
                bid = self.buyer_id_map[inn]
                pid = self.product_id_map[code]
                ts = row['publication_date']
                interactions.append((bid, pid, ts))

        # Deduplicate at (buyer, product) level
        seen = set()
        unique_interactions = []
        for bid, pid, ts in interactions:
            if (bid, pid) not in seen:
                seen.add((bid, pid))
                unique_interactions.append((bid, pid, ts))

        return unique_interactions

    def save(self, path):
        """Save feature engineer state to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load feature engineer state from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def run_pipeline(data_path):
    """Run the full preprocessing pipeline and return all artifacts."""
    print("Loading and cleaning data...")
    df = load_and_clean(data_path)
    print(f"  Cleaned dataset: {len(df)} records")

    print("Splitting data temporally...")
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # If train is very small, use all pre-December data for training
    if len(train_df) < 100:
        print("  Train set too small, merging with validation...")
        train_df = pd.concat([train_df, val_df])
        # Use first 20% of test as validation
        test_sorted = test_df.sort_values('publication_date')
        split_idx = int(len(test_sorted) * 0.2)
        val_df = test_sorted.iloc[:split_idx]
        test_df = test_sorted.iloc[split_idx:]
        print(f"  Adjusted - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Fitting feature engineer...")
    fe = FeatureEngineer()
    fe.fit(pd.concat([train_df, val_df, test_df]))  # Fit encoders on all data for coverage

    print("Building buyer features...")
    buyer_features = fe.build_buyer_features(train_df)
    print(f"  {len(buyer_features)} buyer profiles")

    print("Building product features...")
    product_features = fe.build_product_features(pd.concat([train_df, val_df, test_df]))
    print(f"  {len(product_features)} product profiles")

    print("Building interactions...")
    train_interactions = fe.build_interactions(train_df)
    val_interactions = fe.build_interactions(val_df)
    test_interactions = fe.build_interactions(test_df)
    print(f"  Train interactions: {len(train_interactions)}")
    print(f"  Val interactions: {len(val_interactions)}")
    print(f"  Test interactions: {len(test_interactions)}")

    return {
        'fe': fe,
        'df': df,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'buyer_features': buyer_features,
        'product_features': product_features,
        'train_interactions': train_interactions,
        'val_interactions': val_interactions,
        'test_interactions': test_interactions,
    }


if __name__ == '__main__':
    import os
    data_path = os.path.join(os.path.dirname(__file__), 'b2b_procurement_merged_dataset.xlsx')
    artifacts = run_pipeline(data_path)
    print("\nPipeline complete!")
    print(f"Buyers: {artifacts['fe'].n_buyers}")
    print(f"Products: {artifacts['fe'].n_products}")
    print(f"Regions: {artifacts['fe'].n_regions}")
    print(f"Institution types: {artifacts['fe'].n_inst_types}")
