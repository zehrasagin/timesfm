"""Debug: gradient flow + alpha training check."""
import sys, numpy as np, torch
sys.path.insert(0, "src")
import timesfm
from graph_adapter import TSFMGraphAdapterModel

torch.set_float32_matmul_precision("high")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
tfm.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=1))

names = ["CO1 Comdty","CL1 Comdty","GC1 Comdty","HG1 Comdty","HO1 Comdty",
         "NG1 Comdty","PA1 Comdty","PL1 Comdty","SI1 Comdty","C 1 Comdty"]
model = TSFMGraphAdapterModel(tfm, names, target_idx=0)

# 1. alpha_logit trainable mi?
trainable = model.get_trainable_params()
alpha_param = model.graph_structure.alpha_logit
alpha_in = any(p.data_ptr() == alpha_param.data_ptr() for p in trainable)
print(f"alpha_logit in trainable_params: {alpha_in}")
print(f"alpha_logit.requires_grad: {alpha_param.requires_grad}")
print(f"alpha_logit value: {alpha_param.item():.4f} -> alpha={torch.sigmoid(alpha_param).item():.4f}")

# 2. node_features gradient check
price_hist = np.random.rand(200, 10).astype(np.float32) * 50 + 50
nf = model.node_feature_builder.build_tensor(price_hist)
print(f"\nnode_feats.requires_grad: {nf.requires_grad}  (numpy->torch = detached)")

# 3. forward_cached gradient flow
target_seq = torch.randn(32, 1280, requires_grad=False)
pred = model.forward_cached(target_seq, price_hist)
pred.backward()

print(f"\nalpha_logit.grad: {alpha_param.grad}")
print(f"alpha_logit.grad is None: {alpha_param.grad is None}")

# LearnedAdj grads
la = model.graph_structure.learned_adj
print(f"\nLearnedAdj W_q.grad is None: {la.W_q.weight.grad is None}")
print(f"LearnedAdj W_k.grad is None: {la.W_k.weight.grad is None}")
if la.W_q.weight.grad is not None:
    print(f"LearnedAdj W_q.grad mean: {la.W_q.weight.grad.abs().mean():.8f}")

# GAT grads
print("\nGAT gradients:")
for n, p in model.gat_network.named_parameters():
    g = "None" if p.grad is None else f"{p.grad.abs().mean():.8f}"
    print(f"  {n}: grad_mean={g}")

# CrossAttn grads
print("\nCrossAttn gradients:")
for n, p in model.cross_attention_adapter.named_parameters():
    g = "None" if p.grad is None else f"{p.grad.abs().mean():.8f}"
    print(f"  {n}: grad_mean={g}")

# PredHead grads
print("\nPredHead gradients:")
for n, p in model.prediction_head.named_parameters():
    g = "None" if p.grad is None else f"{p.grad.abs().mean():.8f}"
    print(f"  {n}: grad_mean={g}")
