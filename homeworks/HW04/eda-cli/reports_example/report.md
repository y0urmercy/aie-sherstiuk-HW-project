# EDA Report

## Общая информация
- **Размер данных**: 36 строк, 14 колонок
- **Пропуски**: 5.6%
- **Оценка качества**: 0.776
- **Параметры отчёта**: top_k=10, min_missing_share=0.1

## Качество данных
### Нет колонок с критическим процентом пропусков

### Колонки с большим количеством нулей (>50%):
- `churned`: 66.7% нулей

## Топ категории
### country
- RU: 21 (58.3%)
- KZ: 5 (13.9%)
- BY: 5 (13.9%)
- UA: 5 (13.9%)

### city
- Moscow: 11 (39.3%)
- Saint Petersburg: 3 (10.7%)
- Almaty: 3 (10.7%)
- Minsk: 3 (10.7%)
- Kyiv: 2 (7.1%)
- Astana: 2 (7.1%)
- Novosibirsk: 1 (3.6%)
- Yekaterinburg: 1 (3.6%)
- Kazan: 1 (3.6%)
- Lviv: 1 (3.6%)

### device
- Desktop: 17 (47.2%)
- Mobile: 15 (41.7%)
- Tablet: 4 (11.1%)

### channel
- Organic: 16 (44.4%)
- Ads: 8 (22.2%)
- Referral: 6 (16.7%)
- Email: 6 (16.7%)

### plan
- Basic: 14 (38.9%)
- Free: 12 (33.3%)
- Pro: 10 (27.8%)

## Числовые характеристики
### user_id
- Min: 1001.0
- Max: 1035.0
- Mean: 1018.19
- Std: 10.17

### sessions_last_30d
- Min: 0.0
- Max: 34.0
- Mean: 11.94
- Std: 8.61

### avg_session_duration_min
- Min: 2.0
- Max: 15.2
- Mean: 7.25
- Std: 3.47

### pages_per_session
- Min: 1.0
- Max: 7.5
- Mean: 4.10
- Std: 1.56

### purchases_last_30d
- Min: 0.0
- Max: 4.0
- Mean: 1.14
- Std: 1.13

### revenue_last_30d
- Min: 0.0
- Max: 7000.0
- Mean: 1575.01
- Std: 1815.28

### churned
- Min: 0.0
- Max: 1.0
- Mean: 0.33
- Std: 0.48

### signup_year
- Min: 2018.0
- Max: 2024.0
- Mean: 2020.97
- Std: 1.52

### n_support_tickets
- Min: 0.0
- Max: 5.0
- Mean: 1.08
- Std: 1.20

## Визуализации
### Гистограммы числовых признаков
См. файлы `hist_*.png`

### Boxplot числовых признаков
![Boxplot числовых признаков](numeric_boxplots.png)

### Bar charts категориальных признаков
См. файлы `barchart_*.png`

### Матрица пропусков
![Матрица пропусков](missing_matrix.png)

### Heatmap корреляции
![Heatmap корреляции](correlation_heatmap.png)
