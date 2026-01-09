import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# [新增] 导入指标计算工具
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

TAB20 = np.array(plt.get_cmap('tab20').colors)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reconstruction_loss(pred_pca, target_pca):
    return F.mse_loss(pred_pca, target_pca)


# (kl_rgb_loss 可以删除或保留，反正不用了)

def _z_to_uint8_by_percentile(z_all: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    lo = np.percentile(z_all, p_lo, axis=0, keepdims=True)
    hi = np.percentile(z_all, p_hi, axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-8)
    z01 = np.clip((z_all - lo) / rng, 0.0, 1.0)
    return (np.round(z01 * 255.0)).astype(np.uint8)


def _plot_scatter(coords, colors, save_path, title, point_size=5.0, is_discrete=False):
    plt.figure(figsize=(10, 10), dpi=150)
    order = np.arange(len(coords))
    np.random.shuffle(order)
    coords = coords[order]

    if is_discrete:
        labels = colors[order].astype(int)
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = (labels == lbl)
            c = TAB20[lbl % len(TAB20)]
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[c], s=point_size, label=f"{lbl}", linewidth=0)
        if len(unique_labels) <= 15:
            plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        c = colors[order]
        plt.scatter(coords[:, 0], coords[:, 1], c=c, s=point_size, marker='.', linewidth=0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    if "UMAP" not in title:
        plt.gca().invert_yaxis()

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_visualization(
        full_coords: np.ndarray,
        full_z_rgb: np.ndarray,
        full_h_fuse: np.ndarray,
        epoch: int,
        base_dir: str,
        cluster_k: int,  # KMeans 的 K
        true_labels=None,  # [新增] 真实标签，用于评估
        point_size: float = 5.0
):
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    np.save(os.path.join(epoch_dir, "z_rgb.npy"), full_z_rgb)
    np.save(os.path.join(epoch_dir, "h_fuse.npy"), full_h_fuse)

    # 1. RGB
    rgb_uint8 = _z_to_uint8_by_percentile(full_z_rgb, p_lo=0.5, p_hi=99.5)
    rgb_norm = rgb_uint8 / 255.0
    _plot_scatter(
        full_coords, rgb_norm,
        os.path.join(epoch_dir, "spatial_rgb.png"),
        title=f"Epoch {epoch} | Spatial RGB",
        point_size=point_size, is_discrete=False
    )

    # 2. KMeans
    print(f"  [Vis] Running KMeans (k={cluster_k})...")
    kmeans = KMeans(n_clusters=cluster_k, n_init=3, random_state=42)
    pred_labels = kmeans.fit_predict(full_h_fuse)
    np.save(os.path.join(epoch_dir, "labels.npy"), pred_labels)

    # [新增] 计算指标
    if true_labels is not None:
        # 确保维度匹配
        if len(true_labels) == len(pred_labels):
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            print(f"  [Metrics] Epoch {epoch} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

            # 将指标写入日志文件
            with open(os.path.join(base_dir, "metrics.log"), "a") as f:
                f.write(f"Epoch {epoch}, ARI: {ari:.4f}, NMI: {nmi:.4f}\n")
        else:
            print(f"  [Warning] Labels length mismatch. True: {len(true_labels)}, Pred: {len(pred_labels)}")

    _plot_scatter(
        full_coords, pred_labels,
        os.path.join(epoch_dir, "spatial_cluster.png"),
        title=f"Epoch {epoch} | Spatial Clusters (k={cluster_k})",
        point_size=point_size, is_discrete=True
    )

    # 3. UMAP
    print(f"  [Vis] Running UMAP...")
    if full_h_fuse.shape[1] > 50:
        pca = PCA(n_components=30)
        h_pca = pca.fit_transform(full_h_fuse)
    else:
        h_pca = full_h_fuse

    reducer = umap.UMAP(n_components=2, n_jobs=-1, random_state=42)
    umap_emb = reducer.fit_transform(h_pca)

    _plot_scatter(
        umap_emb, pred_labels,
        os.path.join(epoch_dir, "umap_cluster.png"),
        title=f"Epoch {epoch} | UMAP Latent",
        point_size=2.0, is_discrete=True
    )
