from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: pd.DataFrame,
    high_cardinality_threshold: int = 50,
    zero_ratio_threshold: float = 0.5,
    id_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    и т.п.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    #эвристика 1: Константные колонки
    constant_columns = []
    for col_summary in summary.columns:
        if col_summary.unique == 1:
            constant_columns.append(col_summary.name)

    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns
    flags["n_constant_columns"] = len(constant_columns)

    #эвристика 2: Высококардинальные категориальные признаки
    high_cardinality_cols = []
    for col_summary in summary.columns:
        if (not col_summary.is_numeric and
                col_summary.unique > high_cardinality_threshold):
            high_cardinality_cols.append((col_summary.name, col_summary.unique))

    flags["has_high_cardinality_categoricals"] = len(high_cardinality_cols) > 0
    flags["high_cardinality_columns"] = high_cardinality_cols
    flags["high_cardinality_threshold"] = high_cardinality_threshold

    #эвристика 3: Колонки с большим количеством нулей
    high_zero_ratio_cols = []
    for col_summary in summary.columns:
        if col_summary.is_numeric and col_summary.non_null > 0:
            zero_count = (df[col_summary.name] == 0).sum()
            zero_ratio = zero_count / col_summary.non_null
            if zero_ratio > zero_ratio_threshold:
                high_zero_ratio_cols.append((col_summary.name, zero_ratio))

    flags["has_many_zero_values"] = len(high_zero_ratio_cols) > 0
    flags["high_zero_ratio_columns"] = high_zero_ratio_cols
    flags["zero_ratio_threshold"] = zero_ratio_threshold

    #эвристика 4: Дубликаты в ID колонке
    flags["has_suspicious_id_duplicates"] = False
    flags["id_duplicates_count"] = 0

    if id_column and id_column in df.columns:
        duplicate_count = df[df.duplicated(subset=[id_column], keep=False)].shape[0]
        flags["has_suspicious_id_duplicates"] = duplicate_count > 0
        flags["id_duplicates_count"] = duplicate_count
        flags["id_column_checked"] = id_column

    # Простейший «скор» качества
    score = 1.0
    score -= max_missing_share * 0.3  # чем больше пропусков, тем хуже
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1

    #подсчет штрафов по новым эвристики
    if flags["has_constant_columns"]:
        score -= 0.2 * flags["n_constant_columns"] / summary.n_cols

    if flags["has_high_cardinality_categoricals"]:
        score -= 0.15 * len(flags["high_cardinality_columns"]) / summary.n_cols

    if flags["has_many_zero_values"]:
        score -= 0.1 * len(flags["high_zero_ratio_columns"]) / summary.n_cols

    if flags["has_suspicious_id_duplicates"]:
        score -= 0.25 * flags["id_duplicates_count"] / summary.n_rows

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


def generate_report(
        df: pd.DataFrame,
        summary: DatasetSummary,
        quality_flags: Dict[str, Any],
        title: str = "EDA Report",
        top_k: int = 10,
        min_missing_share: float = 0.1,
        include_boxplots: bool = True,
        include_category_barcharts: bool = True,
) -> str:
    """
    Генерация Markdown отчёта с использованием новых параметров и эвристик.
    """
    report_lines = [
        f"# {title}",
        "",
        "## Общая информация",
        f"- **Размер данных**: {summary.n_rows} строк, {summary.n_cols} колонок",
        f"- **Пропуски**: {quality_flags['max_missing_share']:.1%}",
        f"- **Оценка качества**: {quality_flags['quality_score']:.3f}",
        f"- **Параметры отчёта**: top_k={top_k}, min_missing_share={min_missing_share}",
        "",
        "## Качество данных",
    ]

    problematic_missing = []
    for col in summary.columns:
        if col.missing_share > min_missing_share:
            problematic_missing.append((col.name, col.missing_share))

    if problematic_missing:
        report_lines.append("### Колонки с высоким процентом пропусков:")
        for col, ratio in problematic_missing:
            report_lines.append(f"- `{col}`: {ratio:.1%}")
        report_lines.append("")
    else:
        report_lines.append("### Нет колонок с критическим процентом пропусков")
        report_lines.append("")

    if quality_flags['has_constant_columns']:
        report_lines.append("### Константные колонки:")
        for col in quality_flags['constant_columns']:
            report_lines.append(f"- `{col}` (все значения одинаковые)")
        report_lines.append("")

    if quality_flags['has_high_cardinality_categoricals']:
        report_lines.append(
            f"### Высококардинальные категориальные признаки (>{quality_flags['high_cardinality_threshold']} уникальных):")
        for col, count in quality_flags['high_cardinality_columns']:
            report_lines.append(f"- `{col}`: {count} уникальных значений")
        report_lines.append("")

    if quality_flags['has_many_zero_values']:
        report_lines.append(
            f"### Колонки с большим количеством нулей (>{quality_flags['zero_ratio_threshold']:.0%}):")
        for col, ratio in quality_flags['high_zero_ratio_columns']:
            report_lines.append(f"- `{col}`: {ratio:.1%} нулей")
        report_lines.append("")

    if quality_flags.get('has_suspicious_id_duplicates', False):
        report_lines.append("### Дубликаты в ID колонке:")
        report_lines.append(
            f"- Колонка `{quality_flags['id_column_checked']}`: {quality_flags['id_duplicates_count']} дублирующихся записей")
        report_lines.append("")

    categorical_cols = [col.name for col in summary.columns if not col.is_numeric]
    if len(categorical_cols) > 0:
        report_lines.append("## Топ категории")
        top_cats = top_categories(df, max_columns=top_k, top_k=top_k)
        for col_name, table in top_cats.items():
            report_lines.append(f"### {col_name}")
            for _, row in table.iterrows():
                report_lines.append(f"- {row['value']}: {int(row['count'])} ({row['share']:.1%})")
            report_lines.append("")

    numeric_cols = [col for col in summary.columns if col.is_numeric]
    if numeric_cols:
        report_lines.append("## Числовые характеристики")
        for col in numeric_cols[:top_k]:  # Ограничиваем вывод
            report_lines.append(f"### {col.name}")
            report_lines.append(f"- Min: {col.min}")
            report_lines.append(f"- Max: {col.max}")
            report_lines.append(f"- Mean: {col.mean:.2f}")
            report_lines.append(f"- Std: {col.std:.2f}")
            report_lines.append("")

    report_lines.append("## Визуализации")

    #гистограммы
    report_lines.append("### Гистограммы числовых признаков")
    report_lines.append("См. файлы `hist_*.png`")
    report_lines.append("")

    #boxplot
    if include_boxplots:
        report_lines.append("### Boxplot числовых признаков")
        report_lines.append("![Boxplot числовых признаков](numeric_boxplots.png)")
        report_lines.append("")

    #bar charts для категорий
    if include_category_barcharts:
        report_lines.append("### Bar charts категориальных признаков")
        report_lines.append("См. файлы `barchart_*.png`")
        report_lines.append("")

    #матрица пропусков
    report_lines.append("### Матрица пропусков")
    report_lines.append("![Матрица пропусков](missing_matrix.png)")
    report_lines.append("")

    #heatmap корреляции
    numeric_cols = [col for col in summary.columns if col.is_numeric]
    if len(numeric_cols) >= 2:
        report_lines.append("### Heatmap корреляции")
        report_lines.append("![Heatmap корреляции](correlation_heatmap.png)")
        report_lines.append("")

    return "\n".join(report_lines)

