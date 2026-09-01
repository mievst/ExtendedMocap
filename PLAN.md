# ExtendedMocap — план проекта

Video → 3D skeleton animation pipeline. MediaPipe features → neural networks →
bone-rotation quaternions → CC_Base (Mixamo) character in Blender.

## Цели

- **GitHub** — код, документация, примеры.
- **HuggingFace Datasets** — датасет (пары mediapipe-фичи → кватернионы).
- **HuggingFace Spaces** — Gradio-демо (видео → 3D скелет).
- Лицензия: MIT.

## Архитектура

```
video ──▶ MediaPipe landmarks ──▶ feature CSV ──▶ NN (quaternions) ──▶ Blender CGI
```

1. **Extract** (`src/extended_mocap/extractor.py`) — MediaPipe pose/hand →
   engineered features (distances, angles, body–hand interactions).
2. **Infer** (`src/extended_mocap/inference.py`) — per-segment `NeuralNetwork` /
   `TemporalModel` → quaternions.
3. **Retarget + render** (`src/extended_mocap/blender/`) — CC_Base rig → MP4 +
   per-frame quaternion CSVs.

## Данные: источники

**Источники = 6 blend-файлов в корне репо** (НЕ CSV). CSV — производные,
перегенерируются из blend.

| blend | арматур | actions | CC_Base (101 кость) |
|---|---|---|---|
| `mocap.blend` | 232 | 241 | 165 |
| `mocap_extended.blend` | 84 | 258 | 84 |
| `mocap_accurig.blend` | 83 | 86 | 83 |
| `mocap_mixamo.blend` | 2 | 13 | 2 |
| `mocap_mocca1.blend` | 3 | 177 | 2 |
| `data/mocap.blend` | 1 | 18 | 1 |

- `mocap.blend` — основной, 165 CC_Base арматур = 163 csv-пары, 99 уникальных
  имён движений (нормализованных, без суффиксов `(N)`).
- 101-bone арматуры — fully-retargeted CC_Base (`Armature.NNN` + action
  `Armature.NNN|motion`). Это источник 163 пар.
- 67-bone арматуры — именованные жесты (partial retarget).
- `extended` и `mocca1` дублируют один набор движений между собой.

## motion_id (ключевое правило)

- `frame-range` из имени `test_XXX.RANGE.csv` **НЕ равен** движению.
  - 13 кадровых диапазонов содержат 2 разных движения (не дубли),
    напр. `(0,873) = conference_briefing_m / waiting_talk_295492`.
  - Одно движение может лежать в нескольких диапазонах
    (Pointing — 4, Idle/Acknowledging/Pulling Lever/Thoughtful Head Shake — 2).
- **Правильный motion_id = нормализованное имя движения**:
  `re.sub(r'\s*\(\d+\)\s*$', '', action.split('|')[0])`.
  - `mocap.blend`: 99 уникальных движений по этому правилу.
- Сплит всегда **по движению**, не по анимации (иначе дубли утекают в
  train/test).

## Честная оценка

- Старый `split_anims` делил по анимации → 23/33 test-анимаций разделяли кадр.
  диапазон с train, метрики завышены.
- Фикс: `split_motions` в `src/extended_mocap/evaluation.py` — 0 утечек.
- `score_segment_over_anims(..., reduce="motion")` — per-motion усреднение по
  репам (каждая motion-group = один point).
- CLI: `--motion-split` (default) / `--no-motion-split`.

## Конвейер данных (пересборка из blend)

**Шаг 1 — извлечь кватернионы** (Blender 5.0, headless):
```bash
uv run python scripts/run_extraction.py
```
Каждый из 6 blend → `data/mocap/csv/{armature}.{motion}.csv` +
`data/blend_manifest.json` (арматура → motion_id).

**Шаг 2 — рендер MP4** (опционально, если нет рендеров):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --render --base data/base.blend
```

**Шаг 3 — MediaPipe фичи** (опционально, из рендеров):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --mediapipe
```

**Шаг 4 — сборка манифеста** (без Blender):
```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json
```
→ `data/motion_manifest.json` с motion_id = имя движения.

## Статус

### Сделано
- Motion-split, per-motion метрики, `data/motion_manifest.json` (обновить при
  пересборке — см. конвейер).
- Headless-конвейер: `run_extraction.py`, `extract_quats_from_blends.py`,
  `render_from_quat_csv.py`, `build_dataset.py`.
- Адаптеры датасетов: `dataset_adapters.py` (Mixamo FBX — готов, AMASS SMPL —
  скелет).

### Осталось
- Прогнать пересборку из blend (только пользователь; тяжёлый, Blender headless).
- Пересчитать baseline/temporal на честном сплите (было: GRU < MLP на
  утекающем сплите — проверить).
- Модель: кандидаты Conv1D/TCN (лёгче GRU на 4GB GPU), 6D/ortho6d вместо
  сырых кватернионов.
- Публикация: HF датасет + Spaces demo.