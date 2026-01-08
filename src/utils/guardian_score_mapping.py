import json
import os

def map_value_strict(x):
    """将 0, 0.5, 1 映射为 0/1，None 保留处理"""
    if x is None:
        return None
    return 0 if x == 0 else 1   # 0.5 -> 1, 1 -> 1

def map_value_loose(x):
    """将 0, 0.5, 1 映射为 0/1，None 保留处理"""
    if x is None:
        return None
    return 0 if x in [0, 0.5] else 1   # 0 -> 0, 0.5 -> 0, 1 -> 1

def map_value_exact(x):
    if x is None:
        return None
    elif x == 0:
        return 0
    elif x == 0.5:
        return 1
    elif x == 1:
        return 2

def filter_valid_pairs(preds, labels, score_mode="strict"):
    preds_clean = []
    labels_clean = []

    for p, l in zip(preds, labels):

        if score_mode == "strict":
            # strict 模式下：0→0，0.5→1，1→1
            mp = map_value_strict(p)
            ml = map_value_strict(l)
        elif score_mode == "loose":
            # loose 模式下：0→0，0.5→0，1→1
            mp = map_value_loose(p)
            ml = map_value_loose(l)
        elif score_mode == "exact":
            # exact 模式下：0→0，0.5→0.5，1→1
            mp = map_value_exact(p)
            ml = map_value_exact(l)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        # 跳过 preds 为 None 的情况
        if mp is None or ml is None:
            continue

        preds_clean.append(mp)
        labels_clean.append(ml)

    assert len(preds_clean) == len(labels_clean)

    return preds_clean, labels_clean