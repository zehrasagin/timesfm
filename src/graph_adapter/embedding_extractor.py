"""
TimesFM Embedding Extractor — TSFM-Graph Adapter V2
=====================================================

Frozen TimesFM backbone'dan patch-level embedding çıkarır.
Her asset bağımsız olarak TimesFM'den geçer → E_i ∈ R^{P×D}

Embedding'ler Gated Fusion'da kullanılır:
  - E_target (P, 1280) → GatedGraphFusionAdapter'a girer
  - GNN node features ayrıdır (handcrafted, NodeFeatureBuilder)

Akış:
  Raw time series (T,) per asset
    → Pad to max_context (multiple of patch_len=32)
    → Optional global normalize (ForecastConfig.normalize_inputs=True ise)
    → Patch: (num_patches, 32)
    → RevIN normalize (running stats ile)
    → Tokenizer: (num_patches, 64) → (num_patches, 1280)
    → 20 Transformer katmanı
    → output_embeddings: (num_patches, 1280)  ← BU bizim E_i

Pooling YAPILMAZ — düz patch-level embedding döner.
Tüm işlem torch.no_grad() içinde yapılır (frozen, gradient yok).
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


class TimesFMEmbeddingExtractor:
    """Frozen TimesFM backbone'dan embedding çıkarıcı.

    TimesFM 2.5 modelinin iç yapısına erişerek patch embedding'leri
    elde eder. Modelin ağırlıkları DEĞİŞTİRİLMEZ.

    Args:
        timesfm_model: Initialize edilmiş TimesFM_2p5_200M_torch modeli.
    """

    def __init__(self, timesfm_model, normalize_inputs: Optional[bool] = None):
        self.model = timesfm_model
        self.normalize_inputs = normalize_inputs

        # nn.Module'e eriş (torch.compile wrapping'i handle et)
        module = timesfm_model.model
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # Compiled model'ı unwrap et
        self.module = module

        # Model sabitleri
        # p= patch length, md= model dimension
        self.patch_len = self.module.p       # 32
        self.model_dim = self.module.md      # 1280
        self.device = next(self.module.parameters()).device

    @torch.no_grad()
    def extract_embeddings(
        self,
        time_series_list: List[np.ndarray],
        max_context: int = 1024,
    ) -> torch.Tensor:
        """Birden fazla asset için patch-level embedding çıkar.

        Pooling YAPILMAZ — ham patch embedding'leri döner.

        Args:
            time_series_list: N adet zaman serisi listesi, her biri (T,) shape.
            max_context: Maksimum context uzunluğu (patch_len'in katı olmalı).

        Returns:
            sequence_embeddings: (N, num_patches, 1280) her patch'in embedding'i.
        """
        N = len(time_series_list)
        p = self.patch_len

        # max_context'i patch_len'in katına yuvarla
        # T = P x 32 olmak zorunda, değilse padding ile en yakın üst kata yuvarla.
        if max_context % p != 0:
            max_context = ((max_context // p) + 1) * p

        num_patches = max_context // p

        # ═══ 1. Input Hazırlama: Pad + Mask ═══
        batch_values = []
        batch_masks = []

        for ts in time_series_list:
            ts = np.array(ts, dtype=np.float64)
            ts = np.nan_to_num(ts, nan=0.0)

            if len(ts) >= max_context:
                value = ts[-max_context:].astype(np.float32)
                mask = np.zeros(max_context, dtype=bool)
            else:
                pad_len = max_context - len(ts) # Pad için gereken uzunluk
                value = np.pad(
                    ts.astype(np.float32), (pad_len, 0), constant_values=0.0
                )
                mask = np.array([True] * pad_len + [False] * len(ts)) 

            batch_values.append(value)
            batch_masks.append(mask)

        inputs = torch.tensor(
            np.array(batch_values), dtype=torch.float32, device=self.device
        )
        masks = torch.tensor(
            np.array(batch_masks), dtype=torch.bool, device=self.device
        )

        # ═══ 2. Global Normalization (normalize_inputs=True ile uyumlu) ═══
        if self._should_normalize_inputs():
            inputs = self._apply_global_normalization(inputs)

        # ═══ 3. Patching ═══
        patched_inputs = inputs.reshape(
            N, num_patches, p
        )  # patch'ler ardışık 32 değer içerir.
        patched_masks = masks.reshape(N, num_patches, p)

        # ═══ 4. RevIN Running Statistics ═══
        # TimesFM'in kullandığı aynı Welford tabanlı running stats
        context_mu, context_sigma = self._compute_running_stats(
            patched_inputs, patched_masks, N, num_patches
        )

        # ═══ 5. RevIN Normalize ═══
        normed_inputs = self._revin(
            patched_inputs, context_mu, context_sigma, reverse=False
        )
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # ═══ 6. Forward Through Frozen Model ═══
        was_training = self.module.training
        self.module.eval()
        try:
            (_, output_emb, _, _), _ = self.module(normed_inputs, patched_masks, None)
        finally:
            if was_training:
                self.module.train()
        # output_emb shape: (N, num_patches, 1280)

        return output_emb  # Düdüz patch embedding, pooling yok

    @torch.no_grad()
    def extract_single(
        self,
        time_series: np.ndarray,
        max_context: int = 1024,
    ) -> torch.Tensor:
        """Tek bir asset için patch-level embedding çıkar.

        Returns:
            sequence_embedding: (num_patches, 1280)
        """
        seq_emb = self.extract_embeddings([time_series], max_context)
        return seq_emb[0]

    def _should_normalize_inputs(self) -> bool:
        """Global pre-normalization aktif mi?

        Varsayılan olarak TimesFM'in compile edilmiş ForecastConfig'i takip eder.
        Böylece graph adapter extractor, backbone'un resmi inference yoluyla
        aynı input normalization sırasını kullanır.
        """
        if self.normalize_inputs is not None:
            return self.normalize_inputs

        forecast_config = getattr(self.model, "forecast_config", None)
        return bool(getattr(forecast_config, "normalize_inputs", False))

    def _apply_global_normalization(self, inputs: torch.Tensor) -> torch.Tensor:
        """TimesFM compile yolundaki global normalization'ı uygula.

        Not:
            Bu işlem resmi TimesFM koduyla uyumlu olacak şekilde patching'den
            önce ve tüm padded context üzerinde uygulanır.
        """
        global_mu = torch.mean(inputs, dim=-1, keepdim=True)
        global_sigma = torch.std(inputs, dim=-1, keepdim=True)
        return self._revin(inputs, global_mu, global_sigma, reverse=False)
    
# PATCH i için incremental istatistik hesaplama

# is_legit = ~pmask
#   → Mask True = padded
#   → is_legit True = gerçek veri

# inc_n = geçerli eleman sayısı

#   inc_n = Σ_j 1{patch_ij valid}

# inc_mu = patch içi ortalama

#   inc_mu = (1 / inc_n) * Σ_j x_ij

# inc_var = patch içi varyans (population variance)

#   inc_var = (1 / inc_n) * Σ_j (x_ij - inc_mu)^2

# inc_sigma = sqrt(inc_var)

#   inc_sigma = √(inc_var)

# Not:
#   inc_n == 0 ise:
#       inc_mu = 0
#       inc_var = 0
#       inc_sigma = 0



# ------------------------------------------------------------
# Önceki kümülatif istatistik:

#   n      = önceki toplam örnek sayısı
#   mu     = önceki kümülatif ortalama
#   sigma  = önceki kümülatif std

# Yeni patch istatistiği:

#   inc_n
#   inc_mu
#   inc_sigma

# Yeni toplam örnek sayısı:

#   new_n = n + inc_n

# Yeni ortalama:

#   new_mu = (n * mu + inc_n * inc_mu) / new_n

# Bu şu formüle denktir:

#   μ_new = (Σ_old + Σ_patch) / total_count


    def _compute_running_stats(
        self,
        patched_inputs: torch.Tensor,
        patched_masks: torch.Tensor,
        batch_size: int,
        num_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TimesFM ile birebir aynı RevIN running statistics hesapla.

        Welford's parallel algorithm: Her patch, önceki tüm patch'lerin
        kümülatif istatistiklerini günceller.

        Args:
            patched_inputs: (B, P, patch_len) patch değerleri.
            patched_masks: (B, P, patch_len) True=masked.
            batch_size: B.
            num_patches: P.

        Returns:
            context_mu: (B, P) her patch için kümülatif ortalama.
            context_sigma: (B, P) her patch için kümülatif standart sapma.
        """
        device = patched_inputs.device

        n = torch.zeros(batch_size, device=device)
        mu = torch.zeros(batch_size, device=device)
        sigma = torch.zeros(batch_size, device=device)

        patch_mu_list = []
        patch_sigma_list = []

        for i in range(num_patches):
            patch = patched_inputs[:, i, :]  # (B, patch_len)
            pmask = patched_masks[:, i, :]   # (B, patch_len)

            # TimesFM'in update_running_stats mantığını replicate et
            is_legit = ~pmask  # True = valid değer, False = masked, patch'teki her değerin maskelenip maskelenmediği bilgisi. is_legit = True olan değerler gerçek veriler, False olanlar padding (maskelenmiş) değerlerdir. İstatistik hesaplamalarında sadece is_legit=True olan değerler dikkate alınır.
            inc_n = is_legit.float().sum(dim=-1)  # (B,) her patch'teki valid değer sayısı. Her patch için kaç tane gerçek (masked olmayan) değer olduğunu sayar. Bu, o patch'teki gerçek verilerin sayısını verir.

            # inc_n : Her patch'teki gerçek (masked olmayan) değerlerin sayısı. İstatistik güncellemesinde bu sayı kullanılır, çünkü sadece gerçek veriler istatistiklere katkıda bulunur. Eğer inc_n=0 ise, o patch tamamen maskelenmiş demektir ve istatistik güncellemesi yapılmaz (inc_mu ve inc_var sıfır olur).
            # inc_mu : Her patch'teki gerçek (masked olmayan) değerlerin ortalaması. Bu, o patch'teki gerçek verilerin ortalamasını verir. Eğer inc_n=0 ise, inc_mu sıfır olur çünkü o patch'te gerçek veri yoktur. 
            inc_mu_num = (patch * is_legit.float()).sum(dim=-1)
            inc_n_safe = torch.where(inc_n == 0, torch.ones_like(inc_n), inc_n)
            inc_mu = inc_mu_num / inc_n_safe 
            inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_mu), inc_mu)


            inc_var_num = (
                ((patch - inc_mu.unsqueeze(-1)) ** 2) * is_legit.float() #unsqueeze ile inc_mu'ya son dim ekleyerek (B, 1) yapıyoruz, böylece patch'teki her değerden inc_mu çıkarabiliyoruz. 
            ).sum(dim=-1) # Her patch'teki gerçek (masked olmayan) değerlerin varyansının payını hesaplar. Önce her değerden inc_mu çıkarılır, karesi alınır, sadece gerçek (masked olmayan) değerler dikkate alınarak toplanır. Bu, o patch'teki gerçek verilerin varyansının payını verir. Eğer inc_n=0 ise, inc_var_num sıfır olur çünkü o patch'te gerçek veri yoktur.
            inc_var = inc_var_num / inc_n_safe
            inc_var = torch.where(inc_n == 0, torch.zeros_like(inc_var), inc_var)
            inc_sigma = torch.sqrt(inc_var)

            new_n = n + inc_n
            new_n_safe = torch.where(
                new_n == 0, torch.ones_like(new_n), new_n
            )

            new_mu = (n * mu + inc_mu * inc_n) / new_n_safe # Kümülatif ortalama güncellemesi. Önceki toplam (n * mu) ile yeni patch'in toplamı (inc_mu * inc_n) toplanır, sonra yeni toplam gerçek değer sayısına (new_n) bölünür. Eğer new_n=0 ise, new_mu sıfır olur çünkü o patch'te gerçek veri yoktur.
            new_mu = torch.where(new_n == 0, torch.zeros_like(new_mu), new_mu)

            term1 = n * sigma.pow(2)
            term2 = inc_n * inc_sigma.pow(2)
            term3 = n * (mu - new_mu).pow(2)
            term4 = inc_n * (inc_mu - new_mu).pow(2)

            new_var = (term1 + term2 + term3 + term4) / new_n_safe
            new_var = torch.where(new_n == 0, torch.zeros_like(new_var), new_var)
            new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

            n, mu, sigma = new_n, new_mu, new_sigma
            patch_mu_list.append(mu.clone())
            patch_sigma_list.append(sigma.clone())

        context_mu = torch.stack(patch_mu_list, dim=1)      # (B, P)
        context_sigma = torch.stack(patch_sigma_list, dim=1)  # (B, P)

        return context_mu, context_sigma

    @staticmethod
    def _revin(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Reversible Instance Normalization (TimesFM ile birebir aynı).

        Args:
            x: (B, P, patch_len) patch değerleri.
            mu: (B, P) per-patch running mean.
            sigma: (B, P) per-patch running std.
            reverse: True ise denormalize, False ise normalize.

        Returns:
            Normalize/denormalize edilmiş tensor, aynı shape.
        """
        _TOL = 1e-6

        # Boyut uyumu: mu ve sigma'ya son dim ekle
        if mu.dim() == x.dim() - 1:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        if reverse:
            return x * sigma + mu
        else:
            safe_sigma = torch.where(sigma < _TOL, torch.ones_like(sigma), sigma)
            return (x - mu) / safe_sigma
