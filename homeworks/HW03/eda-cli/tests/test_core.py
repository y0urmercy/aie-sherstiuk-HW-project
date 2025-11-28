from __future__ import annotations

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_constant_columns_detection():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'constant_col': [1, 1, 1, 1],  #константная колонка
        'normal_col': [1, 2, 3, 4],
        'mixed_col': ['A', 'B', 'C', 'D']
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags['has_constant_columns'] == True
    assert 'constant_col' in flags['constant_columns']
    assert flags['n_constant_columns'] == 1


def test_high_cardinality_detection():
    df = pd.DataFrame({
        'id': range(60),
        'high_card_col': [f'category_{i}' for i in range(60)],
        'low_card_col': ['A', 'B'] * 30
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df, high_cardinality_threshold=50)

    assert flags['has_high_cardinality_categoricals'] == True
    assert len(flags['high_cardinality_columns']) == 1
    assert flags['high_cardinality_columns'][0][0] == 'high_card_col'
    assert flags['high_cardinality_columns'][0][1] == 60


def test_many_zeros_detection():
    df = pd.DataFrame({
        'id': range(10),
        'many_zeros': [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
        'few_zeros': [1, 2, 3, 4, 5, 0, 0, 1, 2, 3]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df, zero_ratio_threshold=0.5)

    assert flags['has_many_zero_values'] == True
    assert len(flags['high_zero_ratio_columns']) == 1
    assert flags['high_zero_ratio_columns'][0][0] == 'many_zeros'
    assert flags['high_zero_ratio_columns'][0][1] == 0.7


def test_id_duplicates_detection():
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 4, 2],
        'name': ['A', 'B', 'C', 'D', 'E', 'F'],
        'value': [10, 20, 30, 40, 50, 60]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df, id_column='user_id')

    assert flags['has_suspicious_id_duplicates'] == True
    assert flags['id_duplicates_count'] == 4
    assert flags['id_column_checked'] == 'user_id'


def test_quality_score_includes_new_factors():
    df_bad = pd.DataFrame({
        'constant': [1, 1, 1],
        'with_nulls': [1, None, 3],
        'many_zeros': [0, 0, 1]
    })

    df_good = pd.DataFrame({
        'id': [1, 2, 3],
        'category': ['A', 'B', 'C'],
        'value': [10, 20, 30]
    })

    summary_bad = summarize_dataset(df_bad)
    missing_df_bad = missing_table(df_bad)
    flags_bad = compute_quality_flags(summary_bad, missing_df_bad, df_bad)

    summary_good = summarize_dataset(df_good)
    missing_df_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_df_good, df_good)

    assert flags_bad['quality_score'] < 0.8
    assert flags_good['quality_score'] >= 0.8


def test_no_problems_detected():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'category': ['A', 'B', 'C'],
        'value': [10, 20, 30]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags['has_constant_columns'] == False
    assert flags['has_high_cardinality_categoricals'] == False
    assert flags['has_many_zero_values'] == False
    assert flags['quality_score'] >= 0.8
