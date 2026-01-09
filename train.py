import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入模块 (假设 dataset 和 model 文件名不变)
from dataset import SpatialSequenceDataset
from model import SpatialTransformer
from utils import set_global_seed, save_visualization

# ================= 配置 =================
DEFAULT_PATH = "/home/yangqx/YYY/LLM/dataset/xenium_he_clustered.h5ad"
OUT_DIR = "./output_transformer_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)

    # 数据参数
    parser.add_argument("--k_spatial", type=int, default=6)
    parser.add_argument("--k_feature", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)

    # 模型参数 (调整为讨论后的折中方案)
    parser.add_argument("--dim_gene", type=int, default=313)  # 这里的维度要和 dataset.n_genes 一致
    parser.add_argument("--d_model", type=int, default=256)  # 改为 256
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)  # 6层

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100)  # 这是一个预训练，100轮通常够了
    parser.add_argument("--lr", type=float, default=1e-4)  # Transformer 对 LR 敏感，建议 1e-4
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="随机Mask的比例")

    # Loss 权重
    parser.add_argument("--w_recon", type=float, default=10.0)
    parser.add_argument("--w_vis", type=float, default=1.0)  # 如果视觉不重要，可以设为 0

    return parser.parse_args()


def run_inference(model, dataloader, device):
    """
    全量推理：返回 z (视觉隐变量) 和 h_fuse (聚类特征)
    """
    model.eval()
    all_z = []
    all_h = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            # 数据搬运
            seq_genes = batch['seq_genes'].to(device)
            rel_coords = batch['rel_coords'].to(device)

            # 推理时不 Mask，直接输入全量数据
            out = model(seq_genes, rel_coords)

            # 收集结果
            # Visual Head 输出的是字典 {'mu': ..., 'z': ...}
            # 我们取 mu 作为确定性的视觉表征
            z = out['visual']['mu'].cpu().numpy()

            # 聚类特征
            h = out['h_fuse'].cpu().numpy()

            all_z.append(z)
            all_h.append(h)

    full_z = np.concatenate(all_z, axis=0)
    full_h = np.concatenate(all_h, axis=0)
    return full_z, full_h


def main():
    args = parse_args()
    set_global_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Init] Output Dir: {args.out_dir}")

    # 1. Dataset
    # -----------------------
    dataset = SpatialSequenceDataset(
        h5ad_path=args.path,
        k_spatial=args.k_spatial,
        k_feature=args.k_feature
    )

    # 获取动态的 Cluster 数量
    n_clusters = dataset.n_clusters
    print(f"[Init] Detected Gene Dimension: {dataset.n_genes}")
    print(f"[Init] Detected Ground Truth Clusters: {n_clusters}")

    # 更新参数
    args.dim_gene = dataset.n_genes

    # 提取真实标签 (numpy array) 用于评估
    true_labels = dataset.true_labels

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    infer_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    full_coords = dataset.coords

    # 2. Model
    # --------
    model = SpatialTransformer(
        dim_gene=args.dim_gene,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        k_spatial=args.k_spatial
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. Training Loop
    # ----------------
    print(f"[Train] Start training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_recon_loss = 0
        total_vis_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            # 数据搬运
            seq_genes = batch['seq_genes'].to(device)
            rel_coords = batch['rel_coords'].to(device)
            target_genes = batch['seq_genes'].to(device)
            target_rgb_mu = batch['target_rgb_mu'].to(device)

            # Random Masking
            B, L, D = seq_genes.shape
            mask_matrix = torch.rand(B, L, device=device) < args.mask_ratio
            masked_input = seq_genes.clone()
            masked_input[mask_matrix] = 0.0

            # Forward
            out = model(masked_input, rel_coords)

            # Loss
            pred_center = out['gene_recon']
            true_center = target_genes[:, 0, :]
            loss_r = nn.MSELoss()(pred_center, true_center)

            pred_rgb = out['visual']['mu']
            loss_v = nn.MSELoss()(pred_rgb, target_rgb_mu)

            loss = (args.w_recon * loss_r) + (args.w_vis * loss_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon_loss += loss_r.item()
            total_vis_loss += loss_v.item()

            pbar.set_postfix({'Recon': loss_r.item(), 'Vis': loss_v.item()})

        scheduler.step()

        # Visualization & Eval
        if epoch % 10 == 0 or epoch == 1:
            avg_r = total_recon_loss / len(train_loader)
            avg_v = total_vis_loss / len(train_loader)
            print(f"Epoch {epoch} | Recon: {avg_r:.4f} | Vis: {avg_v:.4f}")

            full_z, full_h = run_inference(model, infer_loader, device)

            # [修改] 传入 cluster_k 和 true_labels
            save_visualization(
                full_coords=full_coords,
                full_z_rgb=full_z,
                full_h_fuse=full_h,
                epoch=epoch,
                base_dir=args.out_dir,
                cluster_k=n_clusters,  # 动态使用真实类别数
                true_labels=true_labels  # 传入真实标签计算 ARI/NMI
            )

            torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pth"))


if __name__ == "__main__":
    main()
