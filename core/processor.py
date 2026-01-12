import numpy as np
import pandas as pd
from umap import UMAP
import hdbscan
from sentence_transformers import SentenceTransformer


class DataProcessor:
    def __init__(self):
        self.local_model = None

    def _load_local_model(self):
        if self.local_model is None:
            print("🔄 Loading local embedding model (all-MiniLM-L6-v2)...")
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_data(self, df, embedding_mode='s2'):
        if df.empty: return df

        # --- 1. Embedding 处理 ---
        if embedding_mode == 's2':
            valid_mask = df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)
            if not valid_mask.all():
                print(f"⚠️ S2 Mode: Dropping {(~valid_mask).sum()} papers without embeddings.")
                df = df[valid_mask].copy()
            if df.empty: return df
            matrix = np.stack(df['embedding'].values)
        else:
            print(f"🧮 Local Mode: Computing embeddings for {len(df)} papers...")
            self._load_local_model()
            texts = (df['title'].fillna("") + ". " + df['abstract'].fillna("")).tolist()
            matrix = self.local_model.encode(texts, show_progress_bar=True)
            df['embedding'] = list(matrix)

        # 参数设置
        n_samples = len(matrix)
        n_neighbors = min(15, n_samples - 1)
        if n_neighbors < 2: n_neighbors = 2

        # --- 2. 聚类专用 UMAP (降维到 10 维) ---
        # 10维足够保留复杂拓扑结构，让 HDBSCAN 更好工作
        print(f"🚀 Step A: Reducing to 10D for Clustering (Input: {matrix.shape})...")
        umap_cluster = UMAP(
            n_neighbors=n_neighbors,
            n_components=10,  # 关键修改：保留更多信息
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        embed_cluster = umap_cluster.fit_transform(matrix)

        # --- 3. HDBSCAN 聚类 ---
        print("🧩 Step B: Running HDBSCAN...")
        # min_samples 设小一点，可以减少被归为噪音(-1)的点
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=17,
            min_samples=2,  # 关键修改：降低噪音容忍度，让更多点归类
            metric='euclidean',
            gen_min_span_tree=True
        )
        cluster_labels = clusterer.fit_predict(embed_cluster)
        df['cluster'] = cluster_labels

        noise_count = (cluster_labels == -1).sum()
        n_clusters = len(set(cluster_labels)) - (1 if noise_count > 0 else 0)
        print(f"✅ Found {n_clusters} clusters. Noise points: {noise_count}/{n_samples}")

        # --- 4. 可视化专用 UMAP (降维到 2 维) ---
        print("🎨 Step C: Reducing to 2D for Visualization...")
        umap_vis = UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.1,  # 稍微分开一点，好看
            metric='cosine',
            random_state=42
        )
        # 注意：这里我们用原始矩阵再跑一次 UMAP 到 2D，通常比 10D->2D 效果更自然
        projections = umap_vis.fit_transform(matrix)

        df['x'] = projections[:, 0]
        df['y'] = projections[:, 1]

        return df