# AGENTS.md

## Правила

- **ЗАПРЕЩЕНО** самостоятельно запускать тренировки моделей (`scripts/build_baseline.py`, `scripts/train_temporal.py`, любые `--epochs`/обучение). Тренировки запускает только пользователь. Ассистент может готовить код, но не инициировать прогон обучения.

## Команды

- Все команды: `uv run python ...`
- Torch с CUDA: `torch==2.13.0+cu132` (index pytorch-cu132 в pyproject/uv.lock). GPU: RTX 3050 Ti 4GB.
- Проверка CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`

## CUDA-тренировки

Запуск обучения на GPU (только пользователь):
```bash
uv run python scripts/train_temporal.py --epochs 30 --window 6 --stride 3 --device cuda
uv run python scripts/build_baseline.py --epochs 50
```

## Данные

- **Источники данных = 6 blend-файлов в корне репо** (НЕ CSV): `mocap.blend` (осн., 165 CC_Base арматур = 163 csv-пары), `mocap_extended` (84), `mocap_accurig` (83), `mocap_mixamo` (2), `mocap_mocca1` (2), `data/mocap` (1). Полный инвентарь — PLAN.md «Данные: источники».
- CSV (mediapipe фичи + кватернионы) — ПРОИЗВОДНЫЕ. Перегенерим из blend, не хранить как основной источник.
- **Грабли motion_id:** frame-range из имени `test_XXX.RANGE.csv` ≠ движение. 13 диапазонов = 2 разных движения, одно движение может быть в нескольких диапазонах (Pointing=4). **Правильный motion_id = нормализованное имя движения**: `re.sub(r'\s*\(\d+\)\s*$','', action.split('|')[0])`. По нему: 99 уникальных движений в mocap.blend.
- Текущая задача данных: извлечь кватернионы из blend по одной арматуре (headless), доработать рендер-файл читать из quat-CSV, собрать датасет с motion_id = имя движения. Тренировки НЕ запускать.

## Сплит/данные

- Сплит **по движению** (`split_motions` в `src/extended_mocap/evaluation.py`), не по анимации — иначе дубли (репы одного движения на NLA-слоях) утекают в train/test. CLI: `--motion-split` (default) / `--no-motion-split`.
- Честная метрика: `score_segment_over_anims(..., reduce="motion")` — per-motion усреднение по репам.
- Манифест движений: `uv run python scripts/make_manifest.py` → `data/motion_manifest.json` (93 движения, 49 с дублями).
- Данные: `data/mediapipe/csv/` (тесты `test_XXX.RANGE.csv`, RANGE = id движения) + `data/mocap/csv/` (кватернионы).

## Конвейер данных (полный цикл пересборки из blend)

Не хранить CSV как основной источник. Не хранить тысячи анимаций в одном `.blend`.

**Шаг 1 — извлечь кватернионы** (Blender 5.0, headless):
```bash
uv run python scripts/run_extraction.py
```
Каждый из 6 blend → `data/mocap/csv/{armature}.{motion}.csv` + `data/blend_manifest.json`.

**Шаг 2 — рендер MP4** (Blender headless, опционально):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --render --base data/base.blend
```
Из quat CSV → `data/renders/{stem}.mp4`.

**Шаг 3 — MediaPipe фичи** (пользователь, опционально):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --mediapipe
```
Из MP4 → `data/mediapipe/csv/{stem}.csv`.

**Шаг 4 — сборка манифеста** (без Blender):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json
```
→ `data/motion_manifest.json` с правильными motion_id ( имя действия, не frame-range).
