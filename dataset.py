import os
import torch
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.sparse as sp

KEY_PCA = 'X_pca'


class SpatialSequenceDataset(Dataset):
    def __init__(
            self,
            h5ad_path: str,
            k_spatial: int = 6,
            k_feature: int = 10,
            k_candidate: int = 100,
    ):
        super().__init__()
        self.path = h5ad_path
        self.k_s = k_spatial
        self.k_f = k_feature
        self.k_c = k_candidate

        print(f"[Dataset] Loading data from {os.path.basename(h5ad_path)}...")
        self.adata = sc.read(h5ad_path)

        # ==========================================
        # [新增] 0. 数据清洗: 过滤 Unlabeled
        # ==========================================
        if 'Cluster' in self.adata.obs:
            print(f"[Dataset] Original cells: {self.adata.n_obs}")
            # 过滤掉 Cluster 列为 'Unlabeled' 的行
            # 注意：这里假设 'Unlabeled' 是字符串。如果是 NaN 需要用 pd.isna 判断
            self.adata = self.adata[self.adata.obs['Cluster'] != 'Unlabeled'].copy()
            print(f"[Dataset] Filtered cells (removed Unlabeled): {self.adata.n_obs}")

            # 保存真实标签供评估使用
            self.true_labels = self.adata.obs['Cluster'].values

            # 自动计算类别数量
            unique_labels = np.unique(self.true_labels)
            self.n_clusters = len(unique_labels)
            print(f"[Dataset] Ground Truth Clusters: {self.n_clusters} classes found.")
        else:
            print("[Warning] 'Cluster' column not found. Cannot filter Unlabeled or calc metrics.")
            self.true_labels = None
            self.n_clusters = 15  # 默认值

        self.n_obs = self.adata.n_obs

        # ==========================================
        # 1. 准备数据矩阵 (逻辑不变，但基于清洗后的数据)
        # ==========================================

        # (A) 空间坐标
        if 'spatial' in self.adata.obsm:
            self.coords = self.adata.obsm['spatial'].astype(np.float32)
        elif 'x_centroid' in self.adata.obs:
            self.coords = self.adata.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
        else:
            raise KeyError("Cannot find coordinates")

        # (B) 基因表达特征 (313 dims)
        if sp.issparse(self.adata.X):
            raw_data = self.adata.X.toarray()
        else:
            raw_data = self.adata.X

        # Log1p
        self.gene_data = np.log1p(raw_data).astype(np.float32)
        self.n_genes = self.gene_data.shape[1]

        # (C) PCA 特征 (仅用于构图)
        if KEY_PCA not in self.adata.obsm:
            # 如果过滤后原来的 PCA 失效了，这里最好重新算一下，或者直接用切片后的
            # 通常切片后直接用旧的 PCA 也是可以的，只要索引对应
            pass
        self.pca_data_for_graph = self.adata.obsm[KEY_PCA].astype(np.float32)

        # (D) RGB 目标 (略，同前)
        try:
            r = self.adata.obs['rgb_mean_R'].values
            g = self.adata.obs['rgb_mean_G'].values
            b = self.adata.obs['rgb_mean_B'].values
            self.rgb_mean = np.stack([r, g, b], axis=1).astype(np.float32)

            vr = self.adata.obs['rgb_var_R'].values
            vg = self.adata.obs['rgb_var_G'].values
            vb = self.adata.obs['rgb_var_B'].values
            self.rgb_var = np.stack([vr, vg, vb], axis=1).astype(np.float32)

            if self.rgb_mean.max() > 1.1:
                self.rgb_mean /= 255.0
                self.rgb_var /= (255.0 ** 2)
        except KeyError:
            self.rgb_mean = np.zeros((self.n_obs, 3), dtype=np.float32)
            self.rgb_var = np.ones((self.n_obs, 3), dtype=np.float32)

        # ==========================================
        # 2. 构建混合图 (重新基于过滤后的数据构建)
        # ==========================================
        self.neighbor_indices = self._build_hybrid_graph()

    # ... 后面的方法 (_build_hybrid_graph, _transform_coords, __getitem__) 保持不变 ...
    def _build_hybrid_graph(self):
        # (代码同上一次回答，此处省略以节省篇幅，直接复制即可)
        # ...
        print(f"[Graph] Building Hybrid Graph (Spatial={self.k_s}, Feature={self.k_f})...")
        nbrs_engine = NearestNeighbors(n_neighbors=self.k_c + 1, algorithm='kd_tree', n_jobs=-1)
        nbrs_engine.fit(self.coords)
        spatial_dists, spatial_indices = nbrs_engine.kneighbors(self.coords)

        final_indices = np.zeros((self.n_obs, 1 + self.k_s + self.k_f), dtype=np.int64)
        final_indices[:, 0] = np.arange(self.n_obs)

        batch_size = 4096
        for i in tqdm(range(0, self.n_obs, batch_size), desc="Computing Neighbors"):
            end = min(i + batch_size, self.n_obs)

            # Spatial
            batch_spatial_nbrs = spatial_indices[i:end, 1: self.k_s + 1]
            final_indices[i:end, 1: 1 + self.k_s] = batch_spatial_nbrs

            # Feature
            center_feats = self.pca_data_for_graph[i:end].reshape(-1, 1, self.pca_data_for_graph.shape[1])
            candidate_idx = spatial_indices[i:end, 1:]
            candidate_feats = self.pca_data_for_graph[candidate_idx]

            center_norm = center_feats / (np.linalg.norm(center_feats, axis=2, keepdims=True) + 1e-8)
            cand_norm = candidate_feats / (np.linalg.norm(candidate_feats, axis=2, keepdims=True) + 1e-8)
            sim_matrix = np.matmul(center_norm, cand_norm.transpose(0, 2, 1)).squeeze(1)

            top_k_args = np.argsort(sim_matrix, axis=1)[:, -self.k_f:]
            top_k_args = np.flip(top_k_args, axis=1)

            rows = np.arange(end - i)[:, None]
            batch_feature_nbrs = candidate_idx[rows, top_k_args]
            final_indices[i:end, 1 + self.k_s:] = batch_feature_nbrs

        return final_indices

    def _transform_coords(self, relative_coords):
        sign = np.sign(relative_coords)
        abs_val = np.abs(relative_coords)
        return sign * np.log1p(abs_val)

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        indices = self.neighbor_indices[idx]
        seq_genes = self.gene_data[indices]
        seq_coords = self.coords[indices]
        center_coord = seq_coords[0:1, :]
        relative_coords = seq_coords - center_coord
        relative_coords_log = self._transform_coords(relative_coords)

        target_rgb_mu = self.rgb_mean[idx]
        target_rgb_var = self.rgb_var[idx]

        return {
            "seq_genes": torch.tensor(seq_genes, dtype=torch.float32),
            "rel_coords": torch.tensor(relative_coords_log, dtype=torch.float32),
            "target_genes": torch.tensor(seq_genes[0], dtype=torch.float32),
            "target_rgb_mu": torch.tensor(target_rgb_mu, dtype=torch.float32),
            "target_rgb_var": torch.tensor(target_rgb_var, dtype=torch.float32),
            "center_idx": torch.tensor(idx, dtype=torch.long)
        }
