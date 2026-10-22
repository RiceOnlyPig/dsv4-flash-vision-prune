#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate token-level MoE routing logs and select frequent experts.

Expected input file names:

    layer_<layer_id>_<global_rank>.txt

Each non-empty line must contain the selected expert IDs for one token, for
example ``124,143,175,153,79,213``.  Files belonging to different ranks are
merged into one counter per model layer.

DeepSeek-V4 example:

    python3 stat_moe.py \
      --top-k 64 \
      --log-dir /path/to/topk_ids_by_layer \
      --start-layer 3 \
      --end-layer 42

By default every rank file is merged.  If inspection proves that all TP ranks
inside a DP replica contain identical logs, ``--dedupe-tp-size 8`` keeps only
global ranks 0, 8, 16, ... so absolute counts are not multiplied by TP size.
Do not enable this option unless that rank layout and duplication are verified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LOG_NAME_RE = re.compile(r"^layer_(?P<layer>\d+)_(?P<rank>\d+)\.txt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并各 rank 的 MoE TopK 日志，并按模型层选出高频专家。"
    )
    parser.add_argument("--top-k", type=int, default=64, help="每层保留的高频专家数。")
    parser.add_argument("--log-dir", required=True, help="layer_<层>_<rank>.txt 所在目录。")
    parser.add_argument(
        "--output",
        default="",
        help="输出 JSON；默认保存到日志目录上一级的 top<N>_experts.json。",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=3,
        help="开始统计的层号；DeepSeek-V4 前三层 hash 路由，默认从 3 开始。",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=42,
        help="最后统计的层号（包含）；DeepSeek-V4 默认 42。",
    )
    parser.add_argument(
        "--experts-per-token",
        type=int,
        default=6,
        help="每行期望的专家 ID 数，DeepSeek-V4 默认 Top6。",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="原模型每层逻辑专家总数，用于检查 ID 范围。",
    )
    parser.add_argument(
        "--dedupe-tp-size",
        type=int,
        default=0,
        metavar="TP",
        help="仅保留 global_rank %% TP == 0 的文件；只有确认 TP 日志重复后才能使用。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0。")
    if args.num_experts <= 0:
        raise ValueError("--num-experts 必须大于 0。")
    if args.top_k > args.num_experts:
        raise ValueError("--top-k 不能大于 --num-experts。")
    if args.experts_per_token <= 0:
        raise ValueError("--experts-per-token 必须大于 0。")
    if args.start_layer < 0 or args.end_layer < args.start_layer:
        raise ValueError("层号范围不合法。")
    if args.dedupe_tp_size < 0:
        raise ValueError("--dedupe-tp-size 不能为负数。")


def discover_files(
    log_dir: Path,
    start_layer: int,
    end_layer: int,
    dedupe_tp_size: int,
) -> tuple[dict[int, list[tuple[int, Path]]], list[str]]:
    files_by_layer: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    ignored_txt: list[str] = []

    for path in sorted(log_dir.glob("*.txt")):
        match = LOG_NAME_RE.fullmatch(path.name)
        if match is None:
            ignored_txt.append(path.name)
            continue

        layer = int(match.group("layer"))
        rank = int(match.group("rank"))
        if layer < start_layer or layer > end_layer:
            continue
        if dedupe_tp_size and rank % dedupe_tp_size != 0:
            continue
        files_by_layer[layer].append((rank, path))

    for entries in files_by_layer.values():
        entries.sort(key=lambda item: item[0])
    return dict(files_by_layer), ignored_txt


def count_file(
    path: Path,
    experts_per_token: int,
    num_experts: int,
) -> tuple[Counter[int], int]:
    counter: Counter[int] = Counter()
    valid_rows = 0

    # utf-8-sig also accepts ordinary UTF-8 and safely strips a possible BOM.
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue

            pieces = [piece.strip() for piece in line.split(",") if piece.strip()]
            if len(pieces) != experts_per_token:
                raise ValueError(
                    f"{path}:{line_number} 应有 {experts_per_token} 个专家 ID，"
                    f"实际为 {len(pieces)}：{line!r}"
                )
            try:
                expert_ids = [int(piece) for piece in pieces]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} 含非整数专家 ID：{line!r}") from exc

            invalid = [expert_id for expert_id in expert_ids if not 0 <= expert_id < num_experts]
            if invalid:
                raise ValueError(
                    f"{path}:{line_number} 专家 ID 超出 [0, {num_experts - 1}]：{invalid}"
                )

            counter.update(expert_ids)
            valid_rows += 1

    if valid_rows == 0:
        raise ValueError(f"日志文件没有有效数据：{path}")
    return counter, valid_rows


def main() -> int:
    args = parse_args()
    validate_args(args)

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"日志目录不存在：{log_dir}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else log_dir.parent / f"top{args.top_k}_experts.json"
    )

    files_by_layer, ignored_txt = discover_files(
        log_dir,
        args.start_layer,
        args.end_layer,
        args.dedupe_tp_size,
    )

    expected_layers = set(range(args.start_layer, args.end_layer + 1))
    actual_layers = set(files_by_layer)
    missing_layers = sorted(expected_layers - actual_layers)
    if missing_layers:
        raise ValueError(f"缺少以下层的日志：{missing_layers}")

    result: dict[str, dict[str, object]] = {}
    total_files = 0
    total_rows = 0

    for layer in sorted(files_by_layer):
        layer_counter: Counter[int] = Counter()
        layer_rows = 0
        ranks: list[int] = []

        for rank, path in files_by_layer[layer]:
            file_counter, file_rows = count_file(
                path,
                args.experts_per_token,
                args.num_experts,
            )
            layer_counter.update(file_counter)
            layer_rows += file_rows
            ranks.append(rank)
            total_files += 1
            total_rows += file_rows

        if len(layer_counter) < args.top_k:
            raise ValueError(
                f"layer_{layer} 只出现过 {len(layer_counter)} 个不同专家，"
                f"不足以选择 Top{args.top_k}。"
            )

        # Stable tie-breaking makes repeated runs reproducible.
        ranked = sorted(layer_counter.items(), key=lambda item: (-item[1], item[0]))
        selected = ranked[: args.top_k]
        result[f"layer_{layer}"] = {
            "topk_ids": [expert_id for expert_id, _ in selected],
            "topk_counts": [count for _, count in selected],
            "total_token_rows": layer_rows,
            "source_ranks": ranks,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print("=" * 72)
    print(f"统计完成：layer_{args.start_layer} 到 layer_{args.end_layer}")
    print(f"模型层数：{len(result)}")
    print(f"读取文件：{total_files}")
    print(f"有效 token 行：{total_rows}")
    print(f"每层选择：Top{args.top_k}")
    print(f"输出文件：{output}")
    if args.dedupe_tp_size:
        print(
            "TP 去重：已启用，只读取 global_rank % "
            f"{args.dedupe_tp_size} == 0 的日志。"
        )
    else:
        print("TP 去重：未启用，已合并全部 rank 文件。")
    if ignored_txt:
        print(f"忽略非标准命名的 txt 文件：{ignored_txt[:10]}")
    print("=" * 72)

    print("各层 Top5 预览：")
    for layer_name, data in result.items():
        ids = data["topk_ids"][:5]
        counts = data["topk_counts"][:5]
        preview = ", ".join(f"{expert_id}({count})" for expert_id, count in zip(ids, counts))
        print(f"  {layer_name}: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

