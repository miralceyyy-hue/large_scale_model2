import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ==========================================
# 1. 辅助组件：相对位置编码器
# ==========================================
class RelativePositionalEncoder(nn.Module):
    """
    接收 Log-Compressed 的相对坐标 (dx, dy)，映射到 d_model。
    注意：输入已经是 dataset 中处理过的 sign * log1p(abs(x))，
    所以在数值上已经归一化，不需要额外的 Scaling。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, rel_coords):
        # rel_coords: (Batch, Seq_Len, 2)
        return self.mlp(rel_coords)


# ==========================================
# 2. 辅助组件：视觉头 (修正版)
# ==========================================
class RenderHeadR(nn.Module):
    """
    负责将隐变量 h 映射到视觉分布参数 (mu, logvar)。
    [修改] 移除了 Sigmoid，不再强制输出 [0,1]，而是输出无界的分布参数。
    """
    def __init__(self, h_dim: int, z_dim: int = 3, hidden: int = 256):
        super().__init__()
        # 加宽了 hidden 层，增强表达能力
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.LayerNorm(hidden), # 加个 Norm 稳一点
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden, 2 * z_dim)  # 输出 mu 和 logvar
        )

    def forward(self, h):
        # h: (Batch, h_dim)
        y = self.net(h)
        mu, logvar = y.chunk(2, dim=-1)

        # 限制 logvar 范围，防止 KL 散度爆炸 (数值稳定性)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        # Reparameterization Trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # [修改] 关键：这里不再使用 Sigmoid
        # 现在的 z 是无界的，符合标准高斯分布假设
        return {"mu": mu, "logvar": logvar, "z": z}


# ==========================================
# 3. 核心模型：Spatial Transformer (扩容版)
# ==========================================
class SpatialTransformer(nn.Module):
    def __init__(
            self,
            dim_gene: int = 313,   # [修改] 输入变为基因数
            d_model: int = 512,    # [修改] 扩大维度
            nhead: int = 8,        # [修改] 增加头数
            num_layers: int = 6,   # [修改] 增加层数
            dim_feedforward: int = 2048, # [修改] FFN 扩大
            dropout: float = 0.1,
            k_spatial: int = 6
    ):
        super().__init__()

        self.d_model = d_model
        self.k_s = k_spatial

        # --- A. 嵌入层 (Embeddings) ---
        # 1. 基因特征投影 (Gene -> d_model)
        # [修改] 加上 LayerNorm，因为 Log1p 后的数据并没有严格归一化到 0-1
        self.input_proj = nn.Sequential(
            nn.Linear(dim_gene, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()  # 加上激活函数，增加一层非线性映射
        )

        # 2. 相对位置编码 (Coords -> d_model)
        self.pos_encoder = RelativePositionalEncoder(d_model)

        # 3. 类型/身份编码
        self.type_embedding = nn.Embedding(3, d_model)

        # --- B. 核心编码器 (Core) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-Norm 结构，深层网络训练更稳
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- C. 输出头 (Heads) ---
        # 1. 重构头 (解码回 Gene Space)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, dim_gene) # 输出维度回其本身
        )

        # 2. 视觉头 (预测 RGB 分布)
        self.render_head = RenderHeadR(h_dim=d_model, z_dim=3)

        self._init_weights()

    def _init_weights(self):
        # 初始化逻辑
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_genes, rel_coords):
        """
        Args:
            seq_genes: (Batch, Seq_Len, 313) - Log1p 后的基因表达
            rel_coords: (Batch, Seq_Len, 2) - Log-Compressed 坐标
        """
        B, L, _ = seq_genes.shape

        # 1. Input Embeddings
        # ------------------------
        # (A) Feature Embedding
        e_feat = self.input_proj(seq_genes)  # (B, L, D)

        # (B) Positional Embedding
        e_pos = self.pos_encoder(rel_coords)  # (B, L, D)

        # (C) Type Embedding
        type_ids = torch.zeros(L, dtype=torch.long, device=seq_genes.device)
        type_ids[1: self.k_s + 1] = 1
        type_ids[self.k_s + 1:] = 2
        e_type = self.type_embedding(type_ids).unsqueeze(0).expand(B, -1, -1)

        # Sum inputs
        x = e_feat + e_pos + e_type

        # 2. Transformer
        # ------------------------
        x_transformed = self.transformer(x)

        # 3. Aggregation (取中心点)
        # ------------------------
        h_center = x_transformed[:, 0, :]  # (B, D)

        # 4. Heads
        # ------------------------
        # (A) Recon: 预测基因
        gene_recon = self.recon_head(h_center)

        # (B) Visual: 预测 RGB 参数
        visual_out = self.render_head(h_center)

        return {
            "h_fuse": h_center,      # 用于聚类的高维特征
            "gene_recon": gene_recon,# (B, 313)
            "visual": visual_out     # mu, logvar, z
        }
