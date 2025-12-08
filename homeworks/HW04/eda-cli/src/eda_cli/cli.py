from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    generate_report,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов с расширенными возможностями")


def _load_csv(
        path: Path,
        sep: str = ",",
        encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
        high_cardinality_threshold: int = typer.Option(50, help="Порог для высококардинальных категорий."),
        zero_ratio_threshold: float = typer.Option(0.5, help="Порог доли нулей для числовых колонок."),
        check_id_column: Optional[str] = typer.Option(None, help="ID колонка для проверки дубликатов."),
) -> None:
    """
    Напечатать краткий обзор датасета с новыми эвристиками качества:
    - размеры;
    - типы;
    - простая табличка по колонкам;
    - флаги качества данных.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    missing_df = missing_table(df)

    #вычисление флагов качества с новыми параметрами
    quality_flags = compute_quality_flags(
        summary, missing_df, df,
        high_cardinality_threshold=high_cardinality_threshold,
        zero_ratio_threshold=zero_ratio_threshold,
        id_column=check_id_column
    )

    summary_df = flatten_summary_for_print(summary)

    typer.echo("=" * 60)
    typer.echo(f"Dataset: {path}")
    typer.echo(f"Строк: {summary.n_rows}, Столбцов: {summary.n_cols}")
    typer.echo(f"Оценка качества: {quality_flags['quality_score']:.3f}")
    typer.echo(f"Макс. доля пропусков: {quality_flags['max_missing_share']:.2%}")
    typer.echo("")

    typer.echo("Флаги качества данных:")
    typer.echo(f"  Слишком мало строк (<100): {quality_flags['too_few_rows']}")
    typer.echo(f"  Слишком много колонок (>100): {quality_flags['too_many_columns']}")
    typer.echo(f"  Слишком много пропусков (>50%): {quality_flags['too_many_missing']}")
    typer.echo(f"  Константные колонки: {quality_flags['has_constant_columns']}")
    if quality_flags['has_constant_columns']:
        typer.echo(f"     Колонки: {', '.join(quality_flags['constant_columns'])}")
    typer.echo(f"  Высококардинальные категории: {quality_flags['has_high_cardinality_categoricals']}")
    if quality_flags['has_high_cardinality_categoricals']:
        for col, count in quality_flags['high_cardinality_columns']:
            typer.echo(f"     {col}: {count} уникальных")
    typer.echo(f"  Много нулевых значений: {quality_flags['has_many_zero_values']}")
    if quality_flags['has_many_zero_values']:
        for col, ratio in quality_flags['high_zero_ratio_columns']:
            typer.echo(f"     {col}: {ratio:.1%} нулей")
    if check_id_column:
        typer.echo(f"  Дубликаты в ID колонке: {quality_flags['has_suspicious_id_duplicates']}")
        if quality_flags['has_suspicious_id_duplicates']:
            typer.echo(f"     {check_id_column}: {quality_flags['id_duplicates_count']} дублирующихся записей")
    typer.echo("=" * 60)


@app.command()
def report(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
        max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
        top_k_categories: int = typer.Option(10, help="Количество топ-категорий для вывода."),
        title: str = typer.Option("EDA Report", help="Заголовок отчёта."),
        min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков для проблемных колонок."),
        high_cardinality_threshold: int = typer.Option(50, help="Порог для высококардинальных категорий."),
        zero_ratio_threshold: float = typer.Option(0.5, help="Порог доли нулей для числовых колонок."),
        check_id_column: Optional[str] = typer.Option(None, help="ID колонка для проверки дубликатов."),
        # НОВЫЙ ПАРАМЕТР: включить boxplot
        include_boxplots: bool = typer.Option(True, help="Включить boxplot для числовых признаков."),
        # НОВЫЙ ПАРАМЕТР: включить bar charts для категорий
        include_category_barcharts: bool = typer.Option(True, help="Включить bar charts для категориальных признаков."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт с новыми параметрами и эвристиками:
    - текстовый overview и summary по колонкам;
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции;
    - расширенные флаги качества данных.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    quality_flags = compute_quality_flags(
        summary, missing_df, df,
        high_cardinality_threshold=high_cardinality_threshold,
        zero_ratio_threshold=zero_ratio_threshold,
        id_column=check_id_column
    )

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    report_content = generate_report(
        df, summary, quality_flags, title, top_k_categories, min_missing_share,
        include_boxplots=include_boxplots,
        include_category_barcharts=include_category_barcharts
    )

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(report_content)

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    if not corr_df.empty:
        plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    if include_boxplots:
        from .viz import plot_numeric_boxplots
        plot_numeric_boxplots(df, out_root / "numeric_boxplots.png", max_columns=8)

    if include_category_barcharts:
        from .viz import plot_top_categories_barchart
        plot_top_categories_barchart(df, out_root, top_k=top_k_categories, max_columns=5)

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    if include_boxplots:
        typer.echo("- Доп. графики: numeric_boxplots.png")
    if include_category_barcharts:
        typer.echo("- Доп. графики: barchart_*.png")
    typer.echo(f"- Использованные параметры: max_hist_columns={max_hist_columns}, top_k={top_k_categories}")


@app.command()
def head(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        n: int = typer.Option(5, help="Количество строк для показа."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Показать первые N строк датасета.
    """
    try:
        df = _load_csv(Path(path), sep=sep, encoding=encoding)
        typer.echo(f"Первые {n} строк файла {path}:")
        typer.echo("=" * 50)
        typer.echo(df.head(n).to_string())
        typer.echo("=" * 50)
    except Exception as e:
        typer.echo(f"Ошибка: {e}")


if __name__ == "__main__":
    app()