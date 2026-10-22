#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely prune routed experts in a DeepSeek-V4-style safetensors checkpoint.

The command is validation-only by default.  Add ``--execute`` only after the
preflight report succeeds and has been reviewed.

Supported checkpoint layouts:

* Per-expert keys, for example ``...mlp.experts.124.gate_proj.weight``.
* Packed expert tensors whose unique expert dimension has size 256.
* Router/gate tensors whose unique expert dimension has size 256.

The leading hash-routed layers are kept intact.  Non-selected experts in later
layers are physically omitted, selected experts are densely re-numbered, and
the same ordering is used when slicing router tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)(?:\.|$)")
MTP_RE = re.compile(r"(?:^|\.)(?:mtp|mtp_layers)\.(?P<mtp>\d+)(?:\.|$)")
INDEXED_EXPERT_RE = re.compile(r"\.experts\.(?P<expert>\d+)(?=\.)")
TOPK_LAYER_RE = re.compile(r"(?:layers\.|layer_)(?P<layer>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 TopN JSON 物理裁剪 DeepSeek-V4 MoE safetensors 权重。"
    )
    parser.add_argument("--model-dir", required=True, help="完整原模型目录，只读。")
    parser.add_argument("--topk-json", required=True, help="stat_moe.py 生成的 TopN JSON。")
    parser.add_argument("--output-dir", required=True, help="全新的输出模型目录。")
    parser.add_argument("--top-k", type=int, default=64, help="后续层保留的专家数。")
    parser.add_argument(
        "--skip-layers",
        default="0,1,2",
        help="不裁剪的 hash 层，逗号分隔；DeepSeek-V4 默认 0,1,2。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正写出裁剪权重；不加时只进行完整预检查。",
    )
    return parser.parse_args()


def parse_skip_layers(value: str) -> set[int]:
    try:
        result = {int(piece.strip()) for piece in value.split(",") if piece.strip()}
    except ValueError as exc:
        raise ValueError(f"--skip-layers 格式错误：{value!r}") from exc
    if any(layer < 0 for layer in result):
        raise ValueError("--skip-layers 不能包含负数。")
    return result


def find_text_config(root: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Find the unique config object containing the MoE topology fields."""
    matches: list[tuple[dict[str, Any], str]] = []
    queue: deque[tuple[Any, str]] = deque([(root, "config")])
    visited: set[int] = set()

    while queue:
        value, path = queue.popleft()
        if not isinstance(value, dict) or id(value) in visited:
            continue
        visited.add(id(value))
        if "num_hidden_layers" in value and "n_routed_experts" in value:
            matches.append((value, path))
        for key, child in value.items():
            if isinstance(child, dict):
                queue.append((child, f"{path}.{key}"))

    if not matches:
        raise ValueError(
            "config.json 中找不到同时包含 num_hidden_layers 和 "
            "n_routed_experts 的配置对象。"
        )
    if len(matches) > 1:
        paths = [path for _, path in matches]
        raise ValueError(f"config.json 中找到多份可能的文本配置，无法安全选择：{paths}")
    return matches[0]


def load_keep_dict(
    json_path: Path,
    num_layers: int,
    old_num_experts: int,
    target_num_experts: int,
    skip_layers: set[int],
) -> dict[int, list[int]]:
    with json_path.open("r", encoding="utf-8-sig") as stream:
        raw = json.load(stream)

    if not isinstance(raw, dict):
        raise ValueError("TopN JSON 顶层必须是 object。")
    if isinstance(raw.get("layers"), dict):
        raw = raw["layers"]

    keep_dict: dict[int, list[int]] = {}
    for key, value in raw.items():
        match = TOPK_LAYER_RE.search(str(key))
        if match is None:
            raise ValueError(f"无法从 TopN JSON 键名解析层号：{key!r}")
        layer = int(match.group("layer"))
        if layer in keep_dict:
            raise ValueError(f"TopN JSON 中 layer_{layer} 重复。")

        if isinstance(value, dict):
            ids = value.get("topk_ids", value.get("keep_experts", value.get("experts")))
        elif isinstance(value, list):
            ids = value
        else:
            ids = None
        if not isinstance(ids, list):
            raise ValueError(f"layer_{layer} 缺少 topk_ids/keep_experts 数组。")

        try:
            experts = [int(item) for item in ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"layer_{layer} 包含非整数专家 ID。") from exc

        if len(experts) != len(set(experts)):
            raise ValueError(f"layer_{layer} 的专家 ID 存在重复。")
        if any(expert < 0 or expert >= old_num_experts for expert in experts):
            raise ValueError(
                f"layer_{layer} 专家 ID 必须位于 [0, {old_num_experts - 1}]。"
            )

        # The frequency order is irrelevant after the router and weights are
        # sliced consistently.  Sorting creates a stable old->new mapping.
        keep_dict[layer] = sorted(experts)

    expected = set(range(num_layers)) - skip_layers
    actual = set(keep_dict) - skip_layers
    missing = sorted(expected - actual)
    unexpected = sorted(layer for layer in actual if layer < 0 or layer >= num_layers)
    if missing:
        raise ValueError(f"TopN JSON 缺少待裁剪层：{missing}")
    if unexpected:
        raise ValueError(f"TopN JSON 含模型范围外的层：{unexpected}")

    for layer in sorted(expected):
        count = len(keep_dict[layer])
        if count != target_num_experts:
            raise ValueError(
                f"layer_{layer} 应包含 {target_num_experts} 个专家，实际为 {count}。"
            )
    return {layer: keep_dict[layer] for layer in sorted(expected)}


def layer_from_key(key: str) -> int | None:
    match = LAYER_RE.search(key)
    return int(match.group("layer")) if match else None


def mtp_from_key(key: str) -> int | None:
    match = MTP_RE.search(key)
    return int(match.group("mtp")) if match else None


def is_routed_expert_key(key: str) -> bool:
    return bool(re.search(r"\.(?:ffn|mlp)\.experts\.", key))


def is_router_key(key: str) -> bool:
    return bool(
        re.search(r"\.(?:ffn|mlp)\.(?:gate|router)\.", key)
        or "e_score_correction_bias" in key
    )


def unique_expert_dim(shape: list[int], old_num_experts: int, key: str) -> int:
    dimensions = [index for index, size in enumerate(shape) if size == old_num_experts]
    if len(dimensions) != 1:
        raise ValueError(
            f"{key} 的形状 {shape} 中应恰好有一个大小为 {old_num_experts} "
            f"的专家维度，实际候选维度为 {dimensions}。"
        )
    return dimensions[0]


def plan_key(
    key: str,
    shape: list[int],
    num_layers: int,
    old_num_experts: int,
    keep_dict: dict[int, list[int]],
    skip_layers: set[int],
    mtp_keep_layer: int,
) -> dict[str, Any]:
    layer = layer_from_key(key)
    mtp_index = mtp_from_key(key)

    # MTP experts are not named model.layers.<N>.  The recipe uses the last
    # main-model layer's selected experts for every MTP layer, as described in
    # the pruning Wiki.  This keeps MTP compatible with the single global
    # n_routed_experts value after pruning.
    if layer is None and mtp_index is not None:
        if is_routed_expert_key(key) or is_router_key(key):
            layer = mtp_keep_layer
        else:
            return {"action": "copy_mtp_aux", "new_key": key, "layer": None, "mtp": mtp_index}

    if layer is None:
        if ".experts." in key or "e_score_correction_bias" in key:
            raise ValueError(f"发现无法解析层号的专家相关权重：{key}")
        return {"action": "copy", "new_key": key, "layer": None}

    if layer < 0 or layer >= num_layers:
        if is_routed_expert_key(key) or is_router_key(key):
            raise ValueError(
                f"发现 num_hidden_layers={num_layers} 范围外的 MoE/MTP 权重，"
                f"当前脚本不会静默处理：{key}"
            )
        return {"action": "copy", "new_key": key, "layer": layer}

    if layer in skip_layers:
        return {"action": "copy_skip_layer", "new_key": key, "layer": layer}

    if layer not in keep_dict:
        raise ValueError(f"内部错误：layer_{layer} 没有保留专家名单。")
    keep = keep_dict[layer]

    indexed_match = INDEXED_EXPERT_RE.search(key) if is_routed_expert_key(key) else None
    if indexed_match is not None:
        old_expert = int(indexed_match.group("expert"))
        if not 0 <= old_expert < old_num_experts:
            raise ValueError(f"{key} 的专家 ID {old_expert} 越界。")
        if old_expert not in keep:
                return {
                "action": "drop_indexed_expert",
                "new_key": None,
                "layer": layer,
                "old_expert": old_expert,
                "mtp": mtp_index,
            }
        new_expert = keep.index(old_expert)
        start, end = indexed_match.span("expert")
        new_key = f"{key[:start]}{new_expert}{key[end:]}"
        return {
            "action": "keep_indexed_expert",
            "new_key": new_key,
            "layer": layer,
            "old_expert": old_expert,
            "new_expert": new_expert,
            "mtp": mtp_index,
        }

    if is_routed_expert_key(key):
        dimension = unique_expert_dim(shape, old_num_experts, key)
        return {
            "action": "slice_packed_expert",
            "new_key": key,
            "layer": layer,
            "dimension": dimension,
            "mtp": mtp_index,
        }

    if is_router_key(key):
        dimensions = [index for index, size in enumerate(shape) if size == old_num_experts]
        if not dimensions:
            return {"action": "copy_router_aux", "new_key": key, "layer": layer}
        if len(dimensions) > 1:
            raise ValueError(
                f"Router 权重 {key} 的形状 {shape} 有多个可能的专家维度：{dimensions}"
            )
        return {
            "action": "slice_router",
            "new_key": key,
            "layer": layer,
            "dimension": dimensions[0],
            "mtp": mtp_index,
        }

    if ".experts." in key:
        raise ValueError(f"发现未识别的专家权重命名，拒绝静默复制：{key}")
    return {"action": "copy", "new_key": key, "layer": layer}


def build_preflight_plan(
    model_dir: Path,
    index: dict[str, Any],
    num_layers: int,
    old_num_experts: int,
    keep_dict: dict[int, list[int]],
    skip_layers: set[int],
) -> tuple[dict[str, dict[str, Any]], Counter[str], dict[int, Counter[str]], dict[int, set[int]]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json 缺少有效 weight_map。")

    file_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[str(filename)].append(str(key))

    plan: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    per_layer: dict[int, Counter[str]] = defaultdict(Counter)
    indexed_ids: dict[int, set[int]] = defaultdict(set)

    for filename, keys in sorted(file_to_keys.items()):
        shard = model_dir / filename
        if not shard.is_file():
            raise FileNotFoundError(f"索引引用的权重分片不存在：{shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            missing_keys = sorted(set(keys) - actual_keys)
            if missing_keys:
                raise ValueError(f"{filename} 缺少索引中的权重，例如：{missing_keys[:5]}")
            for key in keys:
                shape = list(handle.get_slice(key).get_shape())
                item = plan_key(
                    key,
                    shape,
                    num_layers,
                    old_num_experts,
                    keep_dict,
                    skip_layers,
                    num_layers - 1,
                )
                item["filename"] = filename
                item["shape"] = shape
                plan[key] = item
                totals[item["action"]] += 1
                layer = item["layer"]
                if isinstance(layer, int):
                    per_layer[layer][item["action"]] += 1
                if item["action"] in {"keep_indexed_expert", "drop_indexed_expert"}:
                    indexed_ids[layer].add(item["old_expert"])

    for layer in sorted(set(range(num_layers)) - skip_layers):
        actions = per_layer[layer]
        expert_actions = (
            actions["keep_indexed_expert"]
            + actions["drop_indexed_expert"]
            + actions["slice_packed_expert"]
        )
        if expert_actions == 0:
            raise ValueError(f"layer_{layer} 没有识别到任何 routed expert 权重。")
        if actions["slice_router"] == 0:
            raise ValueError(f"layer_{layer} 没有识别到需要裁剪的 Router 专家维度。")
        if indexed_ids[layer] and indexed_ids[layer] != set(range(old_num_experts)):
            missing = sorted(set(range(old_num_experts)) - indexed_ids[layer])
            raise ValueError(
                f"layer_{layer} 的逐专家权重没有覆盖全部 {old_num_experts} 个专家，"
                f"缺少 ID 示例：{missing[:20]}"
            )

    return plan, totals, dict(per_layer), dict(indexed_ids)


def print_preflight(
    model_dir: Path,
    output_dir: Path,
    config_path: str,
    num_layers: int,
    old_num_experts: int,
    target_num_experts: int,
    skip_layers: set[int],
    keep_dict: dict[int, list[int]],
    totals: Counter[str],
    per_layer: dict[int, Counter[str]],
) -> None:
    print("=" * 80)
    print("DeepSeek-V4 MoE 物理裁剪预检查")
    print("=" * 80)
    print(f"原模型：{model_dir}")
    print(f"输出目录：{output_dir}")
    print(f"文本配置位置：{config_path}")
    print(f"模型层数：{num_layers}")
    print(f"原专家数：{old_num_experts}")
    print(f"目标专家数：{target_num_experts}")
    print(f"完整保留层：{sorted(skip_layers)}")
    print(f"裁剪层数：{len(keep_dict)}")
    print(f"权重动作统计：{dict(totals)}")
    print("每层检查摘要：")
    for layer in sorted(keep_dict):
        actions = per_layer[layer]
        print(
            f"  layer_{layer}: keep_indexed={actions['keep_indexed_expert']}, "
            f"drop_indexed={actions['drop_indexed_expert']}, "
            f"packed={actions['slice_packed_expert']}, "
            f"router={actions['slice_router']}"
        )
    print("=" * 80)


def slice_tensor(tensor: torch.Tensor, dimension: int, keep: list[int]) -> torch.Tensor:
    indices = torch.tensor(keep, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, dimension, indices).contiguous()


def write_pruned_checkpoint(
    model_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    text_config: dict[str, Any],
    index: dict[str, Any],
    plan: dict[str, dict[str, Any]],
    keep_dict: dict[int, list[int]],
    target_num_experts: int,
) -> None:
    incomplete = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已经存在，拒绝覆盖：{output_dir}")
    if incomplete.exists():
        raise FileExistsError(
            f"发现上次残留的临时目录，拒绝覆盖：{incomplete}。"
            "请人工确认其内容后再处理。"
        )
    incomplete.mkdir(parents=True)

    weight_map = index["weight_map"]
    file_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[str(filename)].append(str(key))

    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_number, (filename, keys) in enumerate(sorted(file_to_keys.items()), 1):
        source = model_dir / filename
        destination = incomplete / filename
        tensors: dict[str, torch.Tensor] = {}

        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in keys:
                item = plan[key]
                if item["action"] == "drop_indexed_expert":
                    continue
                tensor = handle.get_tensor(key)
                if item["action"] in {"slice_packed_expert", "slice_router"}:
                    layer = item["layer"]
                    tensor = slice_tensor(tensor, item["dimension"], keep_dict[layer])
                new_key = item["new_key"]
                if new_key in tensors or new_key in new_weight_map:
                    raise ValueError(f"裁剪后权重键发生冲突：{new_key}")
                tensors[new_key] = tensor.contiguous()
                new_weight_map[new_key] = filename
                total_size += tensor.numel() * tensor.element_size()

        if not tensors:
            # A shard containing only dropped per-expert tensors legitimately
            # disappears; no new index entry will reference it.
            print(
                f"[{shard_number}/{len(file_to_keys)}] {filename} 裁剪后为空，跳过该分片",
                flush=True,
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(destination))
        print(f"[{shard_number}/{len(file_to_keys)}] 已写入 {destination.name}", flush=True)

    processed_shards = set(file_to_keys)
    for item in model_dir.iterdir():
        if item.name in processed_shards or item.name in {
            "config.json",
            "model.safetensors.index.json",
        }:
            continue
        destination = incomplete / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        elif item.is_file():
            shutil.copy2(item, destination)

    text_config["n_routed_experts"] = target_num_experts
    with (incomplete / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = total_size
    new_index = {"metadata": metadata, "weight_map": new_weight_map}
    with (incomplete / "model.safetensors.index.json").open("w", encoding="utf-8") as stream:
        json.dump(new_index, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    remap = {
        f"layer_{layer}": {
            "new_to_old": keep,
            "old_to_new": {str(old): new for new, old in enumerate(keep)},
        }
        for layer, keep in keep_dict.items()
    }
    with (incomplete / "expert_remap.json").open("w", encoding="utf-8") as stream:
        json.dump(remap, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    os.replace(incomplete, output_dir)
    print(f"裁剪完成：{output_dir}")


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0。")

    model_dir = Path(args.model_dir).expanduser().resolve()
    topk_json = Path(args.topk_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    skip_layers = parse_skip_layers(args.skip_layers)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"原模型目录不存在：{model_dir}")
    if not topk_json.is_file():
        raise FileNotFoundError(f"TopN JSON 不存在：{topk_json}")
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("--output-dir 不能等于原模型目录，也不能位于原模型目录内部。")

    config_file = model_dir / "config.json"
    index_file = model_dir / "model.safetensors.index.json"
    with config_file.open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream)
    with index_file.open("r", encoding="utf-8-sig") as stream:
        index = json.load(stream)

    text_config, config_path = find_text_config(config)
    num_layers = int(text_config["num_hidden_layers"])
    old_num_experts = int(text_config["n_routed_experts"])
    if args.top_k >= old_num_experts:
        raise ValueError(
            f"--top-k={args.top_k} 必须小于原专家数 {old_num_experts}，否则不是裁剪。"
        )
    if any(layer >= num_layers for layer in skip_layers):
        raise ValueError(f"--skip-layers 超出模型层范围 0..{num_layers - 1}。")

    keep_dict = load_keep_dict(
        topk_json,
        num_layers,
        old_num_experts,
        args.top_k,
        skip_layers,
    )
    plan, totals, per_layer, _ = build_preflight_plan(
        model_dir,
        index,
        num_layers,
        old_num_experts,
        keep_dict,
        skip_layers,
    )
    print_preflight(
        model_dir,
        output_dir,
        config_path,
        num_layers,
        old_num_experts,
        args.top_k,
        skip_layers,
        keep_dict,
        totals,
        per_layer,
    )

    if not args.execute:
        print("当前只完成预检查，没有写入任何模型文件。")
        print("确认报告正确后，使用相同命令并增加 --execute 才会真正裁剪。")
        return 0

    write_pruned_checkpoint(
        model_dir,
        output_dir,
        config,
        text_config,
        index,
        plan,
        keep_dict,
        args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
