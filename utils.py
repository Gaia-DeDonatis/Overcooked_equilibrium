import gym
from flask import Flask, jsonify, request
from flask_cors import CORS
from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate
import numpy as np
from stable_baselines3 import PPO
import time

import torch
import torch.nn as nn
from collections import deque
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


import gym_macro_overcooked
import gym as _gym

# 顶部 import 区域补充
import os
import json
import uuid
import datetime as dt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ABIGatedExtractorWithConf(BaseFeaturesExtractor):
    """
    观测末尾 6 维：
      [A, B, I, conf_A, conf_B, conf_I]
      其中 A/B/I ∈ [-1,1]（你现在用的是 ±1）， conf_* ∈ [0,1]（1=很确定, 0=很不确定）

    门控逻辑：
      gate_A_pos = relu(A) * conf_A
      gate_A_neg = relu(-A) * conf_A
      （B/I 同理）
      -> 用上述 gate 缩放共享特征 f，再拼上原始 A/B/I 与 conf_* 作为原始上下文。

    输出维度：6*base_dim + 6
    """
    def __init__(self, observation_space: gym.spaces.Box, base_dim: int = 64, conf_power: float = 1.0):
        self.obs_dim = observation_space.shape[0]
        assert self.obs_dim >= 6, "obs 需要包含至少 6 个 ABI 相关维度: [A,B,I,conf_A,conf_B,conf_I]"
        super().__init__(observation_space, features_dim=6 * base_dim + 6)

        self.base_dim = base_dim
        self.conf_power = conf_power  # conf^gamma，gamma>1时更保守

        # 非 ABI 部分的编码器
        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dim - 6, 256), nn.ReLU(),
            nn.Linear(256, base_dim), nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 切分观测
        x = obs[..., : self.obs_dim - 6]                 # 非 ABI
        A = obs[..., self.obs_dim - 6 : self.obs_dim - 5]
        B = obs[..., self.obs_dim - 5 : self.obs_dim - 4]
        I = obs[..., self.obs_dim - 4 : self.obs_dim - 3]
        cA = obs[..., self.obs_dim - 3 : self.obs_dim - 2].clamp(0.0, 1.0)
        cB = obs[..., self.obs_dim - 2 : self.obs_dim - 1].clamp(0.0, 1.0)
        cI = obs[..., self.obs_dim - 1 : self.obs_dim     ].clamp(0.0, 1.0)

        # 共享特征
        f = self.backbone(x)  # [B, D]

        # 正/负门 + 确定度调制（不确定时减弱门控）
        # 可选对 conf 做指数：conf^gamma，gamma>1→更保守
        # if self.conf_power != 1.0:
        #     cA = cA.pow(self.conf_power)
        #     cB = cB.pow(self.conf_power)
        #     cI = cI.pow(self.conf_power)


        # for continous ABI
        A = torch.where(A >= 0.5, torch.tensor(1.0), torch.tensor(-1.0))
        B = torch.where(B >= 0.5, torch.tensor(1.0), torch.tensor(-1.0))
        I = torch.where(I >= 0.5, torch.tensor(1.0), torch.tensor(-1.0))

        
        A_pos, A_neg = torch.relu(A), torch.relu(-A)
        B_pos, B_neg = torch.relu(B), torch.relu(-B)
        I_pos, I_neg = torch.relu(I), torch.relu(-I)

        # gate_A_pos = A_pos * cA
        # gate_A_neg = A_neg * cA
        # gate_B_pos = B_pos * cB
        # gate_B_neg = B_neg * cB
        # gate_I_pos = I_pos * cI
        # gate_I_neg = I_neg * cI

        gate_A_pos = A_pos
        gate_A_neg = A_neg
        gate_B_pos = B_pos
        gate_B_neg = B_neg
        gate_I_pos = I_pos
        gate_I_neg = I_neg


        # 应用门控（逐元素缩放）
        out = torch.cat([
            f * gate_A_pos, f * gate_A_neg,
            f * gate_B_pos, f * gate_B_neg,
            f * gate_I_pos, f * gate_I_neg,
            A, B, I,       # 原始 ABI 符号（±1）
            cA, cB, cI     # 确定度（0..1）
        ], dim=-1)
        return out
    



# ===== 工具：Beta 与均匀先验的 KL，用于抑制过度自信 =====
def kl_beta_to_uniform(alpha, beta):
    # KL( Beta(a,b) || Beta(1,1) ) = -log B(a,b) + (a-1)psi(a) + (b-1)psi(b) - (a+b-2)psi(a+b)
    lgamma = torch.lgamma
    psi = torch.digamma
    a, b = alpha, beta
    return -(lgamma(a) + lgamma(b) - lgamma(a + b)) + (a - 1) * psi(a) + (b - 1) * psi(b) - (a + b - 2) * psi(a + b)

# ===== 二分类的 Evidential 损失：CE + λ*KL(Beta||Uniform) =====
def evidential_binary_loss(alpha, beta, y, lam=1e-3):
    # y ∈ {0,1}，形状 [B, 1]
    S = alpha + beta
    p = (alpha / S).clamp(1e-6, 1-1e-6)  # Beta 均值作为 P(y=1)
    ce = F.binary_cross_entropy(p, y, reduction='mean')
    kl = kl_beta_to_uniform(alpha, beta).mean()
    loss = ce + lam * kl
    metrics = {'ce': ce.item(), 'kl': kl.item(), 'S_mean': S.mean().item()}
    return loss, metrics



class LightweightTransformerABIModel_adaptivelength(nn.Module):
    """
    自适应长度 + 共享注意力；输出每个维度的 Beta 参数 (alpha, beta)。
    - 预测值：p = alpha / (alpha + beta)
    - 不确定性：S = alpha + beta（越小越不确定），或使用 Beta 方差。
    """
    def __init__(self, state_dim, hidden_dim=32, nhead=2, num_layers=1,
                 lr=1e-3, evidential_lambda=1e-3):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 共享注意力向量
        self.attn_vec = nn.Parameter(torch.randn(hidden_dim))

        # 三个维度的中间头（共享主干，不共享头部）
        self.head_A = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_B = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_I = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())

        # 每个维度输出 Beta 的 alpha / beta（softplus 保正）
        self.fc_A_alpha = nn.Linear(64, 1)
        self.fc_A_beta  = nn.Linear(64, 1)
        self.fc_B_alpha = nn.Linear(64, 1)
        self.fc_B_beta  = nn.Linear(64, 1)
        self.fc_I_alpha = nn.Linear(64, 1)
        self.fc_I_beta  = nn.Linear(64, 1)

        self.evidential_lambda = evidential_lambda
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.data_buffer = []

    @staticmethod
    def _make_pad_mask(lengths, max_len):
        device = lengths.device
        rng = torch.arange(max_len, device=device).unsqueeze(0)  # [1, T]
        pad_mask = rng >= lengths.unsqueeze(1)                   # [B, T], True=padding
        return pad_mask

    @staticmethod
    def _make_recent_window_mask(lengths, max_len, k_recent):
        device = lengths.device
        k = torch.clamp(torch.as_tensor(k_recent, device=device), min=1)
        start = torch.clamp(lengths - k, min=0)                          # [B]
        idx   = torch.arange(max_len, device=device).unsqueeze(0)        # [1, T]
        # 窗口内为 False（保留），其他 True（遮蔽）
        in_window = (idx >= start.unsqueeze(1)) & (idx < lengths.unsqueeze(1))
        mask = ~in_window                                                 # [B, T]
        return mask

    def _attn_pool(self, h, attn_vec, mask_bool, lengths):
        """
        h: [B, T, H]
        attn_vec: [H]
        mask_bool: [B, T] True=mask
        lengths: [B]
        """
        B, T, H = h.shape
        logits = torch.matmul(h, attn_vec)                 # [B, T]
        logits = logits.masked_fill(mask_bool, float('-inf'))

        # 兜底：若某一行全被 mask，则取“最后一个有效 token”
        all_masked = torch.isinf(logits).all(dim=1)        # [B]
        if all_masked.any():
            weights = torch.zeros_like(logits)
            last_idx = (lengths - 1).clamp(min=0)          # [B]
            weights[torch.arange(B, device=h.device), last_idx] = 1.0
        else:
            weights = torch.softmax(logits, dim=1)         # [B, T]

        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled

    def forward(self, x, lengths=None, k_ability=15, k_benevolence=30, k_integrity=30):
        """
        x: [B, T, state_dim]（可包含 padding）
        lengths: [B] 每个序列真实长度；若为 None，默认全长有效
        返回：
          (alpha_A, beta_A), (alpha_B, beta_B), (alpha_I, beta_I) 形状均为 [B, 1]
        """
        B, T, _ = x.shape
        device = x.device
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=device)

        # 1) 输入投影
        x_proj = self.input_proj(x)  # [B, T, H]

        # 2) Transformer 编码（带 padding mask）
        pad_mask = self._make_pad_mask(lengths, T)  # [B, T], True=padding
        h = self.transformer(x_proj, src_key_padding_mask=pad_mask)  # [B, T, H]

        # 3) 三个指标各自“最近 K 步”窗口（与 pad_mask 叠加）
        mask_A = pad_mask | self._make_recent_window_mask(lengths, T, k_ability)
        mask_B = pad_mask | self._make_recent_window_mask(lengths, T, k_benevolence)
        mask_I = pad_mask | self._make_recent_window_mask(lengths, T, k_integrity)

        # 4) 共享注意力向量做池化
        pooled_A = self._attn_pool(h, self.attn_vec, mask_A, lengths)  # [B, H]
        pooled_B = self._attn_pool(h, self.attn_vec, mask_B, lengths)
        pooled_I = self._attn_pool(h, self.attn_vec, mask_I, lengths)

        # 5) 维度头部
        zA = self.head_A(pooled_A)
        zB = self.head_B(pooled_B)
        zI = self.head_I(pooled_I)

        eps = 1e-4
        alpha_A = F.softplus(self.fc_A_alpha(zA)) + eps
        beta_A  = F.softplus(self.fc_A_beta(zA))  + eps
        alpha_B = F.softplus(self.fc_B_alpha(zB)) + eps
        beta_B  = F.softplus(self.fc_B_beta(zB))  + eps
        alpha_I = F.softplus(self.fc_I_alpha(zI)) + eps
        beta_I  = F.softplus(self.fc_I_beta(zI))  + eps

        return (alpha_A, beta_A), (alpha_B, beta_B), (alpha_I, beta_I)

    @staticmethod
    def beta_mean_strength(alpha, beta):
        S = alpha + beta
        p = (alpha / S).clamp(1e-6, 1-1e-6)
        return p, S

    def store_data(self, state_history, ability, benevolence, integrity):
        if isinstance(state_history, torch.Tensor):
            sh = state_history.detach().cpu()
        else:
            sh = torch.as_tensor(state_history)
        self.data_buffer.append((sh, float(ability), float(benevolence), float(integrity)))

    def _collate_batch(self, batch):
        seqs, A, B, I = [], [], [], []
        for sh, a, b, i in batch:
            t = torch.as_tensor(sh, dtype=torch.float32)
            seqs.append(t)
            A.append([a]); B.append([b]); I.append([i])
        lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
        states = pad_sequence(seqs, batch_first=True)  # [B, T_max, D]
        abilitys     = torch.tensor(A, dtype=torch.float32)
        benevolences = torch.tensor(B, dtype=torch.float32)
        integritys   = torch.tensor(I, dtype=torch.float32)
        return states, lengths, abilitys, benevolences, integritys

    def train_step(self, batch_size=None, k_ability=15, k_benevolence=30, k_integrity=30,
                   device=None, loss_weights=(1.0, 1.0, 1.0)):
        if not self.data_buffer:
            return None

        batch = self.data_buffer if batch_size is None else self.data_buffer[:batch_size]
        states, lengths, abilitys, benevolences, integritys = self._collate_batch(batch)

        if device is None:
            device = next(self.parameters()).device
        states       = states.to(device)
        lengths      = lengths.to(device)
        abilitys     = abilitys.to(device)
        benevolences = benevolences.to(device)
        integritys   = integritys.to(device)

        self.optimizer.zero_grad()
        (aA, bA), (aB, bB), (aI, bI) = self.forward(
            states, lengths=lengths,
            k_ability=k_ability, k_benevolence=k_benevolence, k_integrity=k_integrity
        )

        # Evidential 损失（A/B/I 各自）
        lam = self.evidential_lambda
        wA, wB, wI = loss_weights
        loss_A, mA = evidential_binary_loss(aA, bA, abilitys,     lam=lam)
        loss_B, mB = evidential_binary_loss(aB, bB, benevolences, lam=lam)
        loss_I, mI = evidential_binary_loss(aI, bI, integritys,   lam=lam)

        total = wA*loss_A + wB*loss_B + wI*loss_I
        total.backward()
        self.optimizer.step()

        return {
            'total_loss': total.item(),
            'A_ce': mA['ce'], 'A_kl': mA['kl'], 'A_S_mean': mA['S_mean'],
            'B_ce': mB['ce'], 'B_kl': mB['kl'], 'B_S_mean': mB['S_mean'],
            'I_ce': mI['ce'], 'I_kl': mI['kl'], 'I_S_mean': mI['S_mean'],
        }

    def predict_beta_params(self, states_np, lengths_np=None,
                            k_ability=15, k_benevolence=30, k_integrity=30, device=None):
        """
        推理便捷接口：返回每维的 alpha/beta 及 p/S
        states_np: [B, T, D] (np 或 tensor)
        lengths_np: [B] (可选；不传则默认全长有效)
        """
        if not torch.is_tensor(states_np):
            states = torch.tensor(states_np, dtype=torch.float32)
        else:
            states = states_np.float()
        B, T, _ = states.shape

        if lengths_np is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        else:
            lengths = torch.tensor(lengths_np, dtype=torch.long)

        if device is None:
            device = next(self.parameters()).device
        states  = states.to(device)
        lengths = lengths.to(device)

        with torch.no_grad():
            (aA, bA), (aB, bB), (aI, bI) = self.forward(
                states, lengths=lengths,
                k_ability=k_ability, k_benevolence=k_benevolence, k_integrity=k_integrity
            )
            pA, SA = self.beta_mean_strength(aA, bA)
            pB, SB = self.beta_mean_strength(aB, bB)
            pI, SI = self.beta_mean_strength(aI, bI)

        out = {
            'A': {'alpha': aA.squeeze(-1), 'beta': bA.squeeze(-1), 'p': pA.squeeze(-1), 'S': SA.squeeze(-1)},
            'B': {'alpha': aB.squeeze(-1), 'beta': bB.squeeze(-1), 'p': pB.squeeze(-1), 'S': SB.squeeze(-1)},
            'I': {'alpha': aI.squeeze(-1), 'beta': bI.squeeze(-1), 'p': pI.squeeze(-1), 'S': SI.squeeze(-1)},
        }
        return out

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'evidential_lambda': self.evidential_lambda
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.evidential_lambda = checkpoint.get('evidential_lambda', self.evidential_lambda)
        print(f"Model loaded from {path}")
