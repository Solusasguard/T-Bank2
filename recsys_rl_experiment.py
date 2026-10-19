import math
import os
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def set_random_seed(seed: int = 42) -> None:
    """Fix random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 1. DATA LOADING & REALISTIC SYNTHETIC FALLBACK (WITH SHORT-CLIP TRAP)
# =====================================================================

def generate_synthetic_sequential_data(
    num_users: int = 1200,
    num_items: int = 300,
    num_interactions: int = 45000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sequential interaction data where:
    - Users have sequential genre & item transition patterns.
    - Short clips have high watched_pct (easy to finish) but low retention value.
    - Full movies matching user taste have moderate watched_pct but high retention (next session return).
    """
    rng = np.random.default_rng(seed)

    item_ids = np.arange(1, num_items + 1)
    genres = rng.choice(["action", "drama", "comedy", "scifi", "thriller"], size=num_items)
    # 25% of catalog are short clips (duration 5-15 min), 75% are full movies/episodes (60-120 min)
    is_short_clip = rng.choice([1, 0], size=num_items, p=[0.25, 0.75])
    duration_min = np.where(
        is_short_clip == 1,
        rng.integers(5, 16, size=num_items),
        rng.integers(60, 125, size=num_items),
    )
    intrinsic_quality = rng.uniform(0.3, 1.0, size=num_items)

    items_df = pd.DataFrame({
        "item_id": item_ids,
        "genre": genres,
        "is_short_clip": is_short_clip,
        "duration_min": duration_min,
        "quality": intrinsic_quality,
    })

    item_genre = dict(zip(items_df["item_id"], items_df["genre"]))
    item_short = dict(zip(items_df["item_id"], items_df["is_short_clip"]))
    item_qual = dict(zip(items_df["item_id"], items_df["quality"]))

    genre_to_items = {
        g: items_df[items_df["genre"] == g]["item_id"].to_numpy()
        for g in ["action", "drama", "comedy", "scifi", "thriller"]
    }

    rows = []
    base_ts = 1700000000
    avg_seq_len = max(15, num_interactions // num_users)

    for u in range(1, num_users + 1):
        fav_genre = rng.choice(["action", "drama", "comedy", "scifi", "thriller"])
        sec_genre = rng.choice(["action", "drama", "comedy", "scifi", "thriller"])
        seq_len = rng.integers(int(avg_seq_len * 0.6), int(avg_seq_len * 1.4))
        curr_ts = base_ts + int(rng.integers(0, 15 * 86400))

        prev_item = int(rng.choice(genre_to_items[fav_genre]))
        for _ in range(seq_len):
            # Markovian transition: user tends to follow favorite genre or sequential neighbor
            p_roll = rng.random()
            if p_roll < 0.55:
                pool = genre_to_items[fav_genre]
                # Prefer items close in ID to simulate franchise/sequel watching
                weights = np.exp(-0.08 * np.abs(pool - ((prev_item % num_items) + 1)))
                weights /= weights.sum()
                item_id = int(rng.choice(pool, p=weights))
            elif p_roll < 0.80:
                pool = genre_to_items[sec_genre]
                item_id = int(rng.choice(pool))
            else:
                item_id = int(rng.integers(1, num_items + 1))

            g_match = 1.0 if item_genre[item_id] == fav_genre else (0.5 if item_genre[item_id] == sec_genre else 0.0)
            short_flag = item_short[item_id]
            qual = item_qual[item_id]

            # Short clips artificially inflate watched_pct (reward hacking trap),
            # while genre match + high quality drives true user satisfaction and quick return
            mean_watch = 35.0 + 30.0 * g_match + 28.0 * short_flag + 12.0 * qual
            watched_pct = float(np.clip(rng.normal(mean_watch, 14.0), 0.0, 100.0))

            # Time until next interaction: satisfied users on full content return much faster
            return_score = 0.65 * g_match + 0.35 * qual - 0.25 * short_flag + 0.2 * (watched_pct / 100.0)
            returned_quickly = 1 if (return_score + rng.normal(0, 0.22)) > 0.55 else 0
            delta_ts = int(rng.integers(600, 14400)) if returned_quickly else int(rng.integers(86400, 5 * 86400))

            rows.append({
                "user_id": u,
                "item_id": item_id,
                "timestamp": curr_ts,
                "watched_pct": round(watched_pct, 2),
                "returned_quickly": returned_quickly,
            })
            curr_ts += delta_ts
            prev_item = item_id

    interactions_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return interactions_df, items_df


def load_or_create_data(
    data_dir: str = "data",
    max_users: int = 1200,
    max_items: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load KION dataset if present in ./data, otherwise use synthetic sequential generator."""
    inter_path = os.path.join(data_dir, "interactions.csv")
    items_path = os.path.join(data_dir, "items.csv")

    if os.path.exists(inter_path) and os.path.exists(items_path):
        print("Loading real KION dataset from ./data...")
        inter = pd.read_csv(inter_path)
        items = pd.read_csv(items_path)
        if "last_watch_dt" in inter.columns and "timestamp" not in inter.columns:
            inter["timestamp"] = pd.to_datetime(inter["last_watch_dt"]).astype("int64") // 10**9

        top_items = inter["item_id"].value_counts().head(max_items).index
        inter = inter[inter["item_id"].isin(top_items)].copy()
        unique_users = inter["user_id"].unique()
        if len(unique_users) > max_users:
            rng = np.random.default_rng(seed)
            sel_users = set(rng.choice(unique_users, size=max_users, replace=False))
            inter = inter[inter["user_id"].isin(sel_users)].copy()

        inter = inter.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        next_ts = inter.groupby("user_id")["timestamp"].shift(-1)
        inter["returned_quickly"] = ((next_ts - inter["timestamp"]) <= 48 * 3600).astype(int).fillna(0)

        item_map = {old_id: idx + 1 for idx, old_id in enumerate(top_items)}
        inter["item_id"] = inter["item_id"].map(item_map)
        items = items[items["item_id"].isin(top_items)].copy()
        items["item_id"] = items["item_id"].map(item_map)
        if "genres" in items.columns and "genre" not in items.columns:
            items["genre"] = items["genres"].fillna("unk").apply(lambda x: str(x).split(",")[0].strip())
        items["is_short_clip"] = (items.get("for_kids", pd.Series(0, index=items.index)).fillna(0) == 1).astype(int)
        return inter.sort_values("timestamp").reset_index(drop=True), items

    print("KION files not found in ./data. Generating sequential dataset with KION schema...")
    return generate_synthetic_sequential_data(num_users=max_users, num_items=max_items, seed=seed)


# =====================================================================
# 2. TIME-BASED SPLITTING & SEQUENCE DATASETS
# =====================================================================

def build_temporal_splits(
    interactions: pd.DataFrame,
    max_seq_len: int = 20,
) -> Tuple[List[Dict], List[Dict], List[Dict], int, int]:
    """
    Split interactions strictly by global timestamps t_80 and t_90.
    Each sample contains:
    - context: list of item_ids strictly before the target event
    - target_item: ground truth next item_id
    - watched_pct: normalized watch percentage [0, 1]
    - returned_quickly: binary indicator of fast return to service
    """
    t80 = int(interactions["timestamp"].quantile(0.80))
    t90 = int(interactions["timestamp"].quantile(0.90))

    train_samples, val_samples, test_samples = [], [], []

    for uid, group in interactions.sort_values("timestamp").groupby("user_id"):
        items = group["item_id"].to_numpy(dtype=np.int64)
        ts = group["timestamp"].to_numpy(dtype=np.int64)
        watches = (group["watched_pct"].to_numpy(dtype=np.float32) / 100.0).clip(0.0, 1.0)
        rets = group["returned_quickly"].to_numpy(dtype=np.float32)

        for idx in range(3, len(items)):
            context = items[max(0, idx - max_seq_len):idx].tolist()
            sample = {
                "user_id": int(uid),
                "context": context,
                "target_item": int(items[idx]),
                "watched_pct": float(watches[idx]),
                "returned_quickly": float(rets[idx]),
                "future_items": items[idx:min(len(items), idx + 5)].tolist(),
            }
            if ts[idx] < t80:
                train_samples.append(sample)
            elif ts[idx] < t90:
                val_samples.append(sample)
            else:
                test_samples.append(sample)

    # Deduplicate test samples per user (keep user's first post-split evaluation point)
    seen_test_users = set()
    unique_test_samples = []
    for s in test_samples:
        if s["user_id"] not in seen_test_users:
            seen_test_users.add(s["user_id"])
            unique_test_samples.append(s)

    return train_samples, val_samples, unique_test_samples, t80, t90


class SequenceDataset(Dataset):
    """Left-padded sequence dataset for causal GPT-2 training and Reward Model training."""

    def __init__(self, samples: List[Dict], max_seq_len: int = 20):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        ctx = s["context"][-self.max_seq_len:]
        pad_len = self.max_seq_len - len(ctx)
        padded_ctx = [0] * pad_len + ctx
        return (
            torch.tensor(padded_ctx, dtype=torch.long),
            torch.tensor(s["target_item"], dtype=torch.long),
            torch.tensor(s["watched_pct"], dtype=torch.float32),
            torch.tensor(s["returned_quickly"], dtype=torch.float32),
        )


# =====================================================================
# 3. GPT-2 CAUSAL TRANSFORMER FOR NEXT-ITEM PREDICTION
# =====================================================================

class CausalSelfAttentionBlock(nn.Module):
    """Standard GPT-2 Pre-LN Transformer Decoder block with causal attention mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2RecSysModel(nn.Module):
    """GPT-2 architecture adapted for sequential item token generation."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 25,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, num_items + 1, bias=False)
        # Weight tying between input item embeddings and output logits head
        self.lm_head.weight = self.item_emb.weight

    def forward_hidden(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = seq_tokens.size()
        positions = torch.arange(0, seq_len, device=seq_tokens.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.drop(self.item_emb(seq_tokens) + self.pos_emb(positions))

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=seq_tokens.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.ln_f(x)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """Return next-item logits [B, num_items + 1] at the final sequence position."""
        hidden = self.forward_hidden(seq_tokens)
        last_hidden = hidden[:, -1, :]
        logits = self.lm_head(last_hidden)
        logits[:, 0] = -1e9  # Mask padding index 0 from ever being recommended
        return logits


# =====================================================================
# 4. REWARD MODEL (PREDICTING WATCH_PCT & LONG-TERM RETENTION)
# =====================================================================

class UserResponseRewardModel(nn.Module):
    """
    Predicts immediate watch ratio (watched_pct in [0, 1]) and session continuation / return
    probability given user interaction history and candidate item.
    """

    def __init__(self, num_items: int, d_model: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head_watch = nn.Sequential(
            nn.Linear(d_model * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.head_retention = nn.Sequential(
            nn.Linear(d_model * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, context_tokens: torch.Tensor, candidate_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_emb = self.item_emb(context_tokens)
        _, h_n = self.gru(ctx_emb)
        user_state = h_n.squeeze(0)  # [B, D]
        cand_emb = self.item_emb(candidate_items)  # [B, D]

        interaction_vec = torch.cat([user_state, cand_emb, user_state * cand_emb], dim=-1)
        pred_watch = self.head_watch(interaction_vec).squeeze(-1)
        pred_ret = self.head_retention(interaction_vec).squeeze(-1)
        return pred_watch, pred_ret


def train_and_evaluate_reward_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_items: int,
    device: torch.device,
    epochs: int = 6,
) -> Tuple[UserResponseRewardModel, Dict[str, float]]:
    model = UserResponseRewardModel(num_items=num_items, d_model=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for ctx, target, watch, ret in train_loader:
            ctx, target, watch, ret = ctx.to(device), target.to(device), watch.to(device), ret.to(device)
            optimizer.zero_grad()
            p_watch, p_ret = model(ctx, target)
            loss = mse_loss(p_watch, watch) + 0.5 * bce_loss(p_ret, ret)
            loss.backward()
            optimizer.step()

    model.eval()
    val_watch_true, val_watch_pred = [], []
    val_ret_true, val_ret_pred = [], []

    with torch.no_grad():
        for ctx, target, watch, ret in val_loader:
            ctx, target = ctx.to(device), target.to(device)
            p_watch, p_ret = model(ctx, target)
            val_watch_true.append(watch.numpy())
            val_watch_pred.append(p_watch.cpu().numpy())
            val_ret_true.append(ret.numpy())
            val_ret_pred.append(p_ret.cpu().numpy())

    y_w_true = np.concatenate(val_watch_true)
    y_w_pred = np.concatenate(val_watch_pred)
    y_r_true = np.concatenate(val_ret_true)
    y_r_pred = np.concatenate(val_ret_pred)

    metrics = {
        "Watch_MAE": round(float(mean_absolute_error(y_w_true, y_w_pred)), 4),
        "Watch_RMSE": round(float(math.sqrt(mean_squared_error(y_w_true, y_w_pred))), 4),
        "Retention_ROC_AUC": round(float(roc_auc_score(y_r_true, y_r_pred)), 4),
    }
    return model, metrics


# =====================================================================
# 5. BASE GPT-2 TRAINING & RL FINE-TUNING (REINFORCE + KL PENALTY)
# =====================================================================

def train_base_gpt2(
    train_loader: DataLoader,
    num_items: int,
    max_seq_len: int,
    device: torch.device,
    epochs: int = 8,
) -> GPT2RecSysModel:
    model = GPT2RecSysModel(num_items=num_items, max_seq_len=max_seq_len + 5, d_model=64, n_heads=4, n_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for ctx, target, _, _ in train_loader:
            ctx, target = ctx.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(ctx)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
    return model


def finetune_gpt2_with_rl(
    base_model: GPT2RecSysModel,
    reward_model: UserResponseRewardModel,
    train_loader: DataLoader,
    device: torch.device,
    reward_mode: str = "watch_only",
    kl_beta: float = 0.15,
    epochs: int = 4,
    lr: float = 6e-4,
) -> GPT2RecSysModel:
    """
    Fine-tune GPT-2 policy using KL-regularized Policy Gradient (REINFORCE with moving baseline).
    - reward_mode == 'watch_only': optimizes immediate predicted watched_pct.
    - reward_mode == 'long_term': optimizes 0.45 * watched_pct + 0.55 * retention_prob + novelty bonus.
    """
    policy = GPT2RecSysModel(
        num_items=base_model.num_items,
        max_seq_len=base_model.max_seq_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
    ).to(device)
    policy.load_state_dict({k: v.clone() for k, v in base_model.state_dict().items()})
    base_model.eval()
    reward_model.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    running_baseline = 0.5

    for _ in range(epochs):
        policy.train()
        for ctx, _, _, _ in train_loader:
            ctx = ctx.to(device)
            optimizer.zero_grad()

            logits = policy(ctx)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                ref_logits = base_model(ctx)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # Sample actions from current policy distribution
            dist = torch.distributions.Categorical(probs=probs)
            sampled_items = dist.sample()  # [B]
            chosen_log_probs = dist.log_prob(sampled_items)

            with torch.no_grad():
                pred_watch, pred_ret = reward_model(ctx, sampled_items)
                if reward_mode == "watch_only":
                    raw_reward = pred_watch
                else:
                    # Penalize recommending items already present in recent context (repetition penalty)
                    already_in_ctx = (ctx == sampled_items.unsqueeze(1)).any(dim=1).float()
                    novelty_bonus = 1.0 - 0.35 * already_in_ctx
                    raw_reward = (0.45 * pred_watch + 0.55 * pred_ret) * novelty_bonus

                # Analytical token-level KL divergence KL(pi_theta || pi_ref)
                kl_div = torch.sum(probs * (log_probs - ref_log_probs), dim=-1)
                total_reward = raw_reward - kl_beta * kl_div
                running_baseline = 0.9 * running_baseline + 0.1 * float(total_reward.mean().item())
                advantage = total_reward - running_baseline

            # Policy gradient loss + explicit KL regularization term for stability
            pg_loss = -(chosen_log_probs * advantage).mean()
            kl_Exact = torch.sum(probs * (log_probs - ref_log_probs), dim=-1).mean()
            loss = pg_loss + kl_beta * kl_Exact
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

    return policy


# =====================================================================
# 6. EVALUATION PROTOCOL & MULTI-STEP AUTOREGRESSIVE ROLLOUTS
# =====================================================================

def evaluate_policy(
    policy: GPT2RecSysModel,
    reward_model: UserResponseRewardModel,
    test_samples: List[Dict],
    items_df: pd.DataFrame,
    max_seq_len: int,
    device: torch.device,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate ranking relevance (Recall@K, NDCG@K, MAP@K), expected rewards, and diversity."""
    policy.eval()
    reward_model.eval()

    short_map = dict(zip(items_df["item_id"], items_df["is_short_clip"]))
    num_items = policy.num_items

    recalls, ndcgs, aps = [], [], []
    top1_watches, top1_rets, top1_shorts = [], [], []
    recommended_top1_items = []

    with torch.no_grad():
        for s in test_samples:
            ctx = s["context"][-max_seq_len:]
            padded_ctx = [0] * (max_seq_len - len(ctx)) + ctx
            ctx_tensor = torch.tensor([padded_ctx], dtype=torch.long, device=device)

            logits = policy(ctx_tensor).squeeze(0)
            top_k_indices = torch.topk(logits, k=k).indices.cpu().numpy().tolist()

            target = s["target_item"]
            future_set = set(s["future_items"])

            # Next-item Recall@K and NDCG@K
            if target in top_k_indices:
                rank = top_k_indices.index(target)
                recalls.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 2))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

            # MAP@K over user's near-future ground-truth horizon
            hits = 0.0
            sum_precisions = 0.0
            for idx_pos, pred_item in enumerate(top_k_indices):
                if pred_item in future_set:
                    hits += 1.0
                    sum_precisions += hits / (idx_pos + 1.0)
            aps.append(sum_precisions / max(1, min(len(future_set), k)))

            # Evaluate Top-1 recommended action with Reward Model
            top1_item = top_k_indices[0]
            recommended_top1_items.append(top1_item)
            cand_tensor = torch.tensor([top1_item], dtype=torch.long, device=device)
            p_w, p_r = reward_model(ctx_tensor, cand_tensor)
            top1_watches.append(float(p_w.item()))
            top1_rets.append(float(p_r.item()))
            top1_shorts.append(float(short_map.get(top1_item, 0)))

    unique_recommended = len(set(recommended_top1_items))
    coverage = unique_recommended / float(num_items)

    return {
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
        f"MAP@{k}": round(float(np.mean(aps)), 4),
        "Pred_WatchPct": round(float(np.mean(top1_watches) * 100.0), 2),
        "Pred_Retention": round(float(np.mean(top1_rets)), 4),
        "Catalog_Cov": round(coverage, 4),
        "ShortClip_Share": round(float(np.mean(top1_shorts)), 4),
    }


def evaluate_multistep_rollout(
    policy: GPT2RecSysModel,
    reward_model: UserResponseRewardModel,
    test_samples: List[Dict],
    max_seq_len: int,
    device: torch.device,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Generate `horizon` items autoregressively by appending each predicted item to context.
    Measure step-by-step reward, repetition rate, and representation drift from true user state.
    """
    policy.eval()
    reward_model.eval()

    contexts = []
    for s in test_samples:
        ctx = s["context"][-max_seq_len:]
        padded = [0] * (max_seq_len - len(ctx)) + ctx
        contexts.append(padded)

    curr_ctx = torch.tensor(contexts, dtype=torch.long, device=device)
    step_records = []

    with torch.no_grad():
        # Initial user state representation from true history
        base_emb = reward_model.item_emb(curr_ctx)
        _, h_init = reward_model.gru(base_emb)
        init_state = F.normalize(h_init.squeeze(0), dim=-1)

        generated_history = []
        for step in range(1, horizon + 1):
            logits = policy(curr_ctx)
            next_items = torch.argmax(logits, dim=-1)  # Greedy rollout token [B]
            generated_history.append(next_items)

            p_watch, p_ret = reward_model(curr_ctx, next_items)

            # Check if predicted item repeats something already in rolling window
            rep_rate = (curr_ctx == next_items.unsqueeze(1)).any(dim=1).float().mean().item()
            unique_items = len(torch.unique(next_items).cpu().numpy())

            # Append generated item to context window
            curr_ctx = torch.cat([curr_ctx[:, 1:], next_items.unsqueeze(1)], dim=1)

            # Measure cosine drift in user history state
            _, h_curr = reward_model.gru(reward_model.item_emb(curr_ctx))
            curr_state = F.normalize(h_curr.squeeze(0), dim=-1)
            cos_sim = (init_state * curr_state).sum(dim=-1).mean().item()
            context_drift = 1.0 - cos_sim

            step_records.append({
                "Step": step,
                "Mean_Pred_WatchPct": round(float(p_watch.mean().item() * 100.0), 2),
                "Mean_Pred_Retention": round(float(p_ret.mean().item()), 4),
                "Unique_Items_Generated": unique_items,
                "Repeat_In_Context_Rate": round(rep_rate, 4),
                "Context_State_Drift": round(context_drift, 4),
            })

    return pd.DataFrame(step_records)


# =====================================================================
# 7. MAIN EXECUTION
# =====================================================================

def main() -> None:
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 20

    interactions, items_df = load_or_create_data(max_users=1200, max_items=300, seed=42)
    num_items = int(items_df["item_id"].max())
    print(f"Loaded dataset: {interactions['user_id'].nunique()} users, {num_items} items, {len(interactions)} events.")

    train_s, val_s, test_s, t80, t90 = build_temporal_splits(interactions, max_seq_len=max_seq_len)
    print(f"Temporal split samples -> Train: {len(train_s)}, Val: {len(val_s)}, Test (unique users): {len(test_s)}")

    train_loader = DataLoader(SequenceDataset(train_s, max_seq_len), batch_size=128, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_s, max_seq_len), batch_size=256, shuffle=False)

    # 1. Train & Evaluate Reward Model
    print("\n[Step 1] Training User Response Reward Model...")
    reward_model, rm_metrics = train_and_evaluate_reward_model(train_loader, val_loader, num_items, device, epochs=6)
    print(f"Reward Model Validation Metrics: {rm_metrics}")

    # 2. Train Base GPT-2 Next-Item Model
    print("\n[Step 2] Training Base GPT-2 Sequential Recommendation Model...")
    base_gpt2 = train_base_gpt2(train_loader, num_items, max_seq_len, device, epochs=8)

    # 3. Fine-tune GPT-2 with RL under different reward & KL configurations
    print("\n[Step 3] Fine-tuning GPT-2 with RL (3 configurations)...")
    rl_naive = finetune_gpt2_with_rl(
        base_gpt2, reward_model, train_loader, device, reward_mode="watch_only", kl_beta=0.0, epochs=4
    )
    rl_kl_watch = finetune_gpt2_with_rl(
        base_gpt2, reward_model, train_loader, device, reward_mode="watch_only", kl_beta=0.15, epochs=4
    )
    rl_long_term = finetune_gpt2_with_rl(
        base_gpt2, reward_model, train_loader, device, reward_mode="long_term", kl_beta=0.15, epochs=4
    )

    # 4. Compare all models on Test Split
    models = {
        "1. Base GPT-2 (Next-Item MLE)": base_gpt2,
        "2. RL-GPT2 (WatchPct Only, No KL, beta=0.0)": rl_naive,
        "3. RL-GPT2 (WatchPct + KL Penalty, beta=0.15)": rl_kl_watch,
        "4. RL-GPT2 (Long-Term Retention + KL, beta=0.15)": rl_long_term,
    }

    comparison_rows = {}
    for name, m in models.items():
        comparison_rows[name] = evaluate_policy(m, reward_model, test_s, items_df, max_seq_len, device, k=10)

    comp_df = pd.DataFrame(comparison_rows).T
    print("\n================ SINGLE-STEP TEST EVALUATION ================")
    print(comp_df.to_string())

    # 5. Multi-Step Autoregressive Rollout Comparison (5 steps ahead)
    print("\n================ 5-STEP AUTOREGRESSIVE ROLLOUT (BASE GPT-2) ================")
    rollout_base = evaluate_multistep_rollout(base_gpt2, reward_model, test_s, max_seq_len, device, horizon=5)
    print(rollout_base.to_string(index=False))

    print("\n================ 5-STEP AUTOREGRESSIVE ROLLOUT (NAIVE RL, NO KL) ================")
    rollout_naive = evaluate_multistep_rollout(rl_naive, reward_model, test_s, max_seq_len, device, horizon=5)
    print(rollout_naive.to_string(index=False))

    print("\n================ 5-STEP AUTOREGRESSIVE ROLLOUT (LONG-TERM RL + KL) ================")
    rollout_lt = evaluate_multistep_rollout(rl_long_term, reward_model, test_s, max_seq_len, device, horizon=5)
    print(rollout_lt.to_string(index=False))


if __name__ == "__main__":
    main()