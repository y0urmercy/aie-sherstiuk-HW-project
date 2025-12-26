# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 и Домашнего задания 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports \
  --max-hist-columns 6 \
  --top-k-categories 5 \
  --min-missing-share 0.3 \
  --title "Мой EDA отчет"
```

Новые параметры из HW03:

- `--max-hist-columns` – максимальное количество числовых колонок для гистограмм
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков
- `--min-missing-share` – порог доли пропусков для проблемных колонок
- `--title` – заголовок отчёта

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

### Другие CLI команды

```bash
# Обзор датасета
uv run eda-cli overview data/example.csv

# JSON-сводка
uv run eda-cli json-summary data/example.csv
```

### Запуск HTTP API сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Сервис будет доступен по адресу: http://localhost:8000
Документация API (Swagger UI): http://localhost:8000/docs

### Доступные эндпоинты

1. Проверка здоровья сервиса
```http
GET /health
```

2. Оценка качества датасета по агрегированным признакам
```http
POST /quality
Content-Type: application/json

{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.1,
  "numeric_cols": 5,
  "categorical_cols": 5
}
```

3. Оценка качества из CSV файла (с обработкой ошибок HTTP 400)
```http
POST /quality-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

Обрабатываемые ошибки (HTTP 400):
- Неправильный content-type
- Ошибка чтения CSV
- Пустой CSV файл

4. Полный набор флагов качества
```http
POST /quality-flags-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

Возвращает полный набор флагов качества из HW03, включая:
- Константные колонки (все значения одинаковые)
- Высокую кардинальность категориальных признаков
- Дубликаты в ID-колонках
- Много нулевых значений в числовых колонках

Параметры запроса:
- `high_cardinality_threshold` - порог для высококардинальных категориальных признаков (по умолчанию 50)
- `zero_ratio_threshold` - порог для обнаружения колонок с нулями (по умолчанию 0.5)
- `id_column` - имя колонки для проверки дубликатов ID

5. Сводка по датасету
```http
POST /summary-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?example_values_per_column=3
```

6. Топ категорий для категориальных признаков
```http
POST /top-categories-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?max_columns=5&top_k=5
```

7. Матрица корреляций Пирсона
```http
POST /correlation-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
```

8. Первые N строк датасета
```http
POST /head-from-csv
Content-Type: multipart/form-data

file: [ваш_csv_файл.csv]
?n=10
```

9. Метрики работы сервиса
```http
GET /metrics
```

### Примеры использования API через curl

Проверка здоровья:
```bash
curl -X GET "http://localhost:8000/health"
```

Оценка качества CSV:
```bash
curl -X POST "http://localhost:8000/quality-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

Получение флагов качества
```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

С параметрами для эвристик:
```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv?high_cardinality_threshold=30&zero_ratio_threshold=0.3" \
  -F "file=@data/example.csv"
```

### Новые эвристики качества данных

В реализации используются следующие эвристики:
- Константные колонки - колонки, где все значения одинаковые
- Высококардинальные категориальные признаки - категориальные колонки с большим количеством уникальных значений
- Колонки с большим количеством нулей - числовые колонки, где более 50% значений равны 0
- Дубликаты в ID колонках - проверка на дублирующиеся значения в ID-колонках

## Тесты

```bash
uv run pytest -q
```

### Структура проекта

```text
src/eda_cli/
├── __init__.py
├── core.py                # Основная логика EDA (новые эвристики из HW03)
├── viz.py                 # Визуализации
├── cli.py                 # CLI интерфейс
└── api.py                 # HTTP API (FastAPI) - HW04

data/
└── example.csv            # Пример данных для тестирования

tests/
└── test_core.py           # Тесты для ядра EDA
```

### Зависимости

Основные зависимости:
- `pandas` - обработка данных
- `matplotlib` - визуализации
- `typer` - CLI интерфейс
- `fastapi` - HTTP API (HW04)
- `uvicorn[standard]` - ASGI сервер (HW04)
- `python-multipart` - обработка загрузки файлов (HW04)
- `pydantic` - валидация данных

### Особенности реализации 

1. Обработка ошибок HTTP 400: Все эндпоинты, принимающие CSV, возвращают 400 при:
  - Неправильном формате файла
  - Ошибках чтения CSV
  - Пустых данных
2. Дополнительный эндпоинт: POST /quality-flags-from-csv использует все новые эвристики из HW03
3. Логирование: Каждый запрос логируется с временем выполнения (latency_ms)
4. Метрики: Эндпоинт /metrics предоставляет статистику работы сервиса
