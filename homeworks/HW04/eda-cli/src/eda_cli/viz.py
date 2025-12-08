from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
        df: pd.DataFrame,
        out_dir: PathLike,
        max_columns: int = 6,
        bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, name in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(s.values, bins=bins)
        ax.set_title(f"Histogram of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        fig.tight_layout()

        out_path = out_dir / f"hist_{i + 1}_{name}.png"
        fig.savefig(out_path)
        plt.close(fig)

        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty dataset", ha="center", va="center")
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(min(12, df.shape[1] * 0.4), 4))
        ax.imshow(mask, aspect="auto", interpolation="none")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("Missing values matrix")
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(10, corr.shape[1]), min(8, corr.shape[0])))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Correlation heatmap")
        fig.colorbar(im, ax=ax, label="Pearson r")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_top_categories_tables(
        top_cats: Dict[str, pd.DataFrame],
        out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths


def plot_numeric_boxplots(
        df: pd.DataFrame,
        out_path: PathLike,
        max_columns: int = 8,
) -> Path:
    """
    Создает boxplot для нескольких числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No numeric columns for boxplots",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        numeric_df = numeric_df.iloc[:, :max_columns]

        fig, ax = plt.subplots(figsize=(max(10, numeric_df.shape[1] * 1.2), 8))

        boxplot = ax.boxplot(
            [numeric_df[col].dropna().values for col in numeric_df.columns],
            labels=numeric_df.columns,
            patch_artist=True,
            showfliers=True
        )

        colors = plt.cm.Set3(np.linspace(0, 1, len(numeric_df.columns)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title("Boxplots of Numeric Columns", fontsize=14, fontweight='bold')
        ax.set_ylabel("Values", fontsize=12)
        ax.set_xlabel("Columns", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_top_categories_barchart(
        df: pd.DataFrame,
        out_dir: PathLike,
        top_k: int = 10,
        max_columns: int = 5,
) -> List[Path]:
    """
    Создает bar chart для топ-N категорий категориальных признаков.
    """
    out_dir = _ensure_dir(out_dir)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    paths: List[Path] = []

    for i, col_name in enumerate(categorical_cols[:max_columns]):
        value_counts = df[col_name].value_counts().head(top_k)

        if value_counts.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(value_counts)), value_counts.values,
                      color=plt.cm.viridis(np.linspace(0, 1, len(value_counts))))

        ax.set_title(f"Top {top_k} Categories in {col_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Categories")
        ax.set_ylabel("Count")

        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')

        for bar, count in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=9)

        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        out_path = out_dir / f"barchart_{col_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        paths.append(out_path)

    return paths