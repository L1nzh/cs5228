from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/supervised/metrics.csv")
    parser.add_argument("--out", default="artifacts/supervised/metrics_barplot.png")
    parser.add_argument("--variant", default="tuned", choices=["baseline", "tuned"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    df = df[(df["variant"] == args.variant) & (df["split"] == args.split)].copy()
    if df.empty:
        raise ValueError(f"no rows found for variant={args.variant}, split={args.split}")

    value_cols = ["accuracy", "precision", "recall", "f1"]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long_df = df.melt(
        id_vars=["model", "variant", "split"],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )

    known_order = ["logreg", "knn", "rf", "hgb", "et"]
    present = [m for m in known_order if m in long_df["model"].unique().tolist()]
    rest = sorted([m for m in long_df["model"].unique().tolist() if m not in present])
    model_order = present + rest
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    metric_order = ["accuracy", "precision", "recall", "f1"]
    long_df["metric"] = pd.Categorical(long_df["metric"], categories=metric_order, ordered=True)
    long_df = long_df.sort_values(["model", "metric"])

    plt.figure(figsize=(9, 4))
    sns.barplot(data=long_df, x="model", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.title(f"Supervised metrics ({args.variant}, {args.split})")
    plt.ylabel("score")
    plt.xlabel("model")
    plt.legend(title="metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote plot to: {out_path}")


if __name__ == "__main__":
    main()
