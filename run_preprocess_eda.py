from __future__ import annotations

import argparse
from pathlib import Path

from src.data import assert_same_schema, load_csv
from src.eda import (
    EDAConfig,
    churn_overview,
    correlation_analysis,
    missing_value_table,
    numeric_summary,
    plot_univariate_distributions,
    top_features_by_mutual_info,
    write_summary,
)
from src.preprocess import fit_transform, infer_feature_types, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="churn-bigml-80.csv")
    parser.add_argument("--test", default="churn-bigml-20.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--area-code-as-categorical", action="store_true", default=True)
    parser.add_argument("--no-area-code-as-categorical", action="store_false", dest="area_code_as_categorical")
    parser.add_argument("--top-n-state", type=int, default=15)
    args = parser.parse_args()

    train = load_csv(args.train)
    test = load_csv(args.test)
    assert_same_schema(train.X, test.X)

    out_root = Path(args.out)
    preprocess_dir = out_root / "preprocess"
    eda_dir = out_root / "eda"

    artifacts = fit_transform(
        train.X,
        train.y,
        test.X,
        test.y,
        area_code_as_categorical=args.area_code_as_categorical,
    )
    save_artifacts(artifacts, preprocess_dir)

    categorical_cols, numeric_cols = infer_feature_types(
        train.X, area_code_as_categorical=args.area_code_as_categorical
    )

    overview = churn_overview(train.y, eda_dir)
    missing = missing_value_table(train.X)
    desc = numeric_summary(train.X, numeric_cols)
    plots = plot_univariate_distributions(
        train.X,
        train.y,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        out_dir=eda_dir,
        top_n_state=EDAConfig(top_n_state=args.top_n_state).top_n_state,
    )
    corr_pairs, corr_plot = correlation_analysis(train.X, numeric_cols, eda_dir)
    if corr_plot is not None:
        plots.append(corr_plot)

    top_features = top_features_by_mutual_info(artifacts.X_train, artifacts.y_train)
    summary_path = eda_dir / "eda_summary.md"
    write_summary(
        summary_path,
        overview=overview,
        missing=missing,
        numeric_desc=desc,
        corr_pairs=corr_pairs,
        top_features=top_features,
        plots=sorted(set(plots)),
    )

    print(f"Wrote preprocess artifacts to: {preprocess_dir}")
    print(f"Wrote EDA outputs to: {eda_dir}")
    print(f"Wrote EDA summary to: {summary_path}")


if __name__ == "__main__":
    main()

