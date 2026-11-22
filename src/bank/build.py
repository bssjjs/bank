"""口径层构建入口：python -m bank.build

读原始库 → 套口径规则 → 自检 → 落 model_dataset.parquet 与 feature_spec.json。
"""

from __future__ import annotations

import sys

import pandas as pd

from bank import dataset as D
from bank import schema as S


def _fmt(facts: dict) -> str:
    added = facts["n_cols_out"] - facts["n_cols_raw"]
    dataset_line = f"输出数据集      {facts['n_cols_out']} 列（新增 {added} 列，未删原始列）"
    target_line = (
        f"目标            {S.TARGET}：正例 {facts['n_positive']:,} 条，占比 {facts['pos_rate']:.2%}"
    )
    lines = [
        "",
        "口径层构建完成",
        "-" * 46,
        f"原始表          {S.RAW_TABLE}（{facts['n_cols_raw']} 列 / {facts['n_rows']:,} 行）",
        dataset_line,
        target_line,
        f"是否被联系过    {facts['n_contacted']:,} 条为 1，{facts['n_never_contacted']:,} 条为 0",
        f"教育程度未知    {facts['n_edu_unknown']:,} 条（走哑变量，不占顺序位）",
        "",
        "距上次联系天数 分档分布",
    ]
    for label, n in facts["pdays_bucket_counts"].items():
        lines.append(f"  {label:<10}{n:>8,}")
    lines += [
        "",
        f"特征数 A 版 {len(D.model_features('A'))} 列 / B 版 {len(D.model_features('B'))} 列",
        f"排除列 {list(S.EXCLUDE_COLS)}",
        "-" * 46,
    ]
    return "\n".join(lines)


def main() -> int:
    raw = D.load_raw()
    df = D.build_dataset(raw)
    facts = D.validate(df, raw)
    spec = D.build_spec(df, facts)
    path, spec_path = D.write_outputs(df, spec)

    # 回读校验：parquet 往返后列、行、标签合计都不能变
    back = pd.read_parquet(path)
    D.validate(back, raw)

    print(_fmt(facts))
    print(f"{path.relative_to(S.ROOT)}  {path.stat().st_size / 1024:.0f} KB")
    print(f"{spec_path.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
