"""Quick import & sanity test for graph_adapter modules."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# 1. Graph Structure
print("Testing graph_structure...")
from graph_adapter.graph_structure import (
    CorrelationAdjacency, SectorAdjacency, SupplyChainAdjacency,
    LearnedAdjacency, HybridGraphStructure,
    COMMODITY_SECTORS, SUPPLY_CHAIN_EDGES,
)
print(f"  Sectors: {list(COMMODITY_SECTORS.keys())}")
print(f"  Supply chain edges: {len(SUPPLY_CHAIN_EDGES)}")

assets = [
    "CO1 Comdty", "CL1 Comdty", "GC1 Comdty", "HG1 Comdty", "HO1 Comdty",
    "NG1 Comdty", "PA1 Comdty", "PL1 Comdty", "SI1 Comdty", "C 1 Comdty",
]
N = len(assets)

# Test adjacency matrices
sector_adj = SectorAdjacency(assets)
A_sector = sector_adj.compute()
print(f"  Sector adj shape: {A_sector.shape}, sum: {A_sector.sum():.0f}")

supply_adj = SupplyChainAdjacency(assets)
A_supply = supply_adj.compute()
print(f"  Supply adj shape: {A_supply.shape}, sum: {A_supply.sum():.0f}")

corr_adj = CorrelationAdjacency(window=30, threshold=0.3)
fake_prices = np.random.randn(100, N).astype(np.float32)
A_corr = corr_adj.compute(fake_prices)
print(f"  Corr adj shape: {A_corr.shape}")

# 2. Learned Adjacency
print("\nTesting learned adjacency...")
learned = LearnedAdjacency(embed_dim=1280, key_dim=64)
fake_emb = torch.randn(N, 1280)
A_learned = learned(fake_emb)
print(f"  Learned adj shape: {A_learned.shape}, range: [{A_learned.min():.3f}, {A_learned.max():.3f}]")

# 3. Hybrid Graph Structure
print("\nTesting hybrid graph structure...")
hybrid = HybridGraphStructure(assets, embed_dim=1280, initial_alpha=0.7)
A_hybrid = hybrid(fake_emb, price_history=fake_prices)
print(f"  Hybrid adj shape: {A_hybrid.shape}")
print(f"  Alpha: {hybrid.alpha.item():.4f}")

# 4. GAT
print("\nTesting GAT...")
from graph_adapter.gat_layer import GATLayer, GATNetwork
gat_net = GATNetwork(embed_dim=1280, graph_dim=256, num_heads=4, num_layers=2)
graph_ctx = gat_net(fake_emb, A_hybrid)
print(f"  GAT output shape: {graph_ctx.shape}")  # (N, 1280)

# 5. Cross-Attention Adapter
print("\nTesting Cross-Attention Adapter...")
from junk.cross_attention_adapter import CrossAttentionAdapter
adapter = CrossAttentionAdapter(embed_dim=1280, adapter_dim=256, num_heads=4)
fake_temporal = torch.randn(32, 1280)  # 32 patches
enhanced = adapter(fake_temporal, graph_ctx)
print(f"  Adapter input:  {fake_temporal.shape}")
print(f"  Adapter output: {enhanced.shape}")
diff = (enhanced - fake_temporal).abs().mean().item()
print(f"  Mean diff (should be ~0 at init): {diff:.6f}")

# 6. Prediction Head
print("\nTesting Prediction Head...")
from graph_adapter.tsfm_graph_model import PredictionHead
head = PredictionHead(embed_dim=1280, hidden_dim=256)
pred = head(enhanced)
print(f"  Prediction shape: {pred.shape}")

# 7. Dataset
print("\nTesting Dataset...")
from graph_adapter.dataset import MultiAssetDataset, collate_fn
import pandas as pd

fake_df = pd.DataFrame(
    np.random.randn(2000, N).cumsum(axis=0) + 50,
    columns=assets,
)
dataset = MultiAssetDataset(fake_df, "CO1 Comdty", assets, context_length=512)
print(f"  Dataset length: {len(dataset)}")
sample = dataset[0]
print(f"  Sample context series: {len(sample['context_series'])} assets, "
      f"each shape {sample['context_series'][0].shape}")
print(f"  Target: {sample['target']:.4f}")

# 8. Parameter counting
print("\nTesting parameter counting...")
trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
trainable += sum(p.numel() for p in gat_net.parameters() if p.requires_grad)
trainable += sum(p.numel() for p in adapter.parameters() if p.requires_grad)
trainable += sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"  Total trainable params (excl. backbone): {trainable:,}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
