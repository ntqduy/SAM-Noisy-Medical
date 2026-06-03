# Project Code Ideas and Overview

## 1. One-Sentence Idea

Dự án này là một benchmark đánh giá độ robust của các mô hình họ SAM trên ảnh y tế khi ảnh đầu vào bị nhiễu có kiểm soát, với cùng một giao thức về dataset, prompt, noise severity, metric, aggregation và visualization.

Tên ý tưởng có thể dùng trong paper:

> A unified robustness benchmark for SAM-based medical image segmentation under controlled image degradations.

## 2. Research Motivation

Các mô hình Segment Anything và các biến thể y tế như MedSAM, SAM-Med2D, MedSAM2, MedSAM3, UltraSAM thường được báo cáo tốt trên ảnh sạch. Tuy nhiên, ảnh y tế thực tế dễ bị suy giảm chất lượng do noise, blur, contrast/brightness shift, compression artifact, pixelation hoặc đặc thù modality như Rician/Poisson noise.

Dự án này đặt câu hỏi:

1. Mô hình nào giữ performance tốt nhất khi ảnh bị nhiễu tăng dần từ L0 đến L9?
2. Loại nhiễu nào phá segmentation mạnh nhất?
3. Prompt mode nào ổn định hơn: point, box hay point+box?
4. Các model y tế có robust hơn SAM gốc hay không?
5. Robustness có nhất quán giữa các dataset/modalities không?

## 3. Core Contribution

Project có thể được mô tả bằng 5 đóng góp chính:

1. Unified benchmark: cùng pipeline cho nhiều SAM-family models, nhiều datasets, nhiều prompt modes.
2. Controlled noise protocol: 12 loại nhiễu với 10 mức severity L0-L9.
3. Deterministic inference: noise sinh on-the-fly nhưng có seed ổn định theo dataset, image_id, noise type và level.
4. Non-oracle candidate mask selection: nếu model trả nhiều mask, final mask được chọn bằng predicted IoU/confidence score của model, không dùng GT metric.
5. Full analysis suite: raw CSV, aggregated stats, robustness statistics, ranking, heatmaps, line plots, qualitative galleries.

## 4. Important Terminology

### Oracle Prompt

Prompt trong benchmark hiện tại là GT-derived prompt. Nghĩa là point và bbox được sinh từ ground-truth mask để chuẩn hóa input prompt giữa các model.

Điều này nên nói rõ trong paper:

> We use GT-derived prompts to isolate model robustness from prompt-generation variability.

### Non-Oracle Mask Selection

Mask selection là non-oracle vì candidate mask không được chọn bằng Dice/IoU thật với GT. Các wrapper gọi model với `multimask_output=True`, sau đó chọn mask dựa trên predicted IoU/confidence score.

Câu nên dùng:

> Given GT-derived prompts, candidate mask selection is oracle-free: when multiple masks are returned, the final mask is selected using the model's predicted IoU/confidence score, without using ground-truth masks or evaluation metrics.

### Evaluation Metric

GT mask chỉ được dùng sau khi prediction xong để tính IoU, Dice, Recall, Precision, F1 và HD.

## 5. High-Level Pipeline

Entry point chính là `main.py`.

Pipeline có 3 stage:

1. Stage 1 `run`: chạy inference dưới nhiễu, ghi raw CSV và artifact ảnh.
2. Stage 1b `aggregate`: aggregate raw CSV thành stats CSV và merged statistics.
3. Stage 2 `visualize`: tạo figures, heatmaps, rankings, galleries.

Lệnh full:

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

Lệnh chạy từng stage:

```bash
python main.py --config configs/full_benchmark.yaml --stage run
python main.py --config configs/full_benchmark.yaml --stage aggregate
python main.py --config configs/full_benchmark.yaml --stage visualize
```

## 6. Stage 1 Detailed Flow

Stage 1 nằm chủ yếu trong `core/experiment_engine.py`.

Flow logic:

1. Load config từ YAML.
2. Build dataset bằng `datasets.dataset_registry.build_dataset`.
3. Build model runner bằng `core.model_manager.ModelManager`.
4. Lặp qua:
   - dataset
   - model
   - prompt_mode
   - noise_type
   - noise_level
   - noise_seed
   - image
5. Đọc ảnh sạch và GT mask.
6. Apply noise on-the-fly bằng `NoiseManager`.
7. Tạo prompt bằng `resolve_prompt`.
8. Chạy `runner.predict(noisy_image, prompt)`.
9. Convert output về binary mask.
10. Tính metric với GT.
11. Ghi row vào raw CSV.
12. Lưu artifact ảnh nếu bật `save_artifacts`.

Pseudocode:

```text
for dataset in datasets:
  for model in models:
    for prompt_mode in prompt_modes:
      load model once
      for noise_type in noise_types:
        for level in L0..L9:
          for seed in noise_seeds:
            for sample in dataset:
              noisy = apply_noise(image, noise_type, level, seed)
              prompt = resolve_prompt(gt_mask, prompt_mode)
              pred_mask = model.predict(noisy, prompt)
              metrics = compute(pred_mask, gt_mask)
              write raw csv row
```

## 7. Main Modules

### `main.py`

Vai trò:

- Parse CLI arguments.
- Load config qua `ConfigManager`.
- Canonicalize output path.
- Chạy stage `run`, `aggregate`, `visualize`, hoặc `all`.
- Hỗ trợ filter dataset/model.
- Hỗ trợ single GPU hoặc multi-GPU model split.
- Tự tìm legacy output path như `outputs/outputs/full_benchmark`.

CLI quan trọng:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 50
python main.py --config configs/full_benchmark.yaml --stage run --datasets BUSI,CAMUS
python main.py --config configs/full_benchmark.yaml --stage run --models SAM2,MedSAM3
python main.py --config configs/full_benchmark.yaml --stage run --device cuda:1
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

### `core/config_manager.py`

Vai trò:

- Load YAML.
- Validate các section bắt buộc: `exp`, `datasets`, `models`.
- Normalize `exp.out_root`, `exp.name`, `noise_config`.
- Resolve device từ `device`, `devices`, hoặc `num_gpus`.

### `core/model_manager.py`

Vai trò:

- Registry/factory cho các model wrappers.
- Map tên model sang class runner.
- Load weights qua `runner.load_model()`.
- Kiểm tra real model đã load chưa.
- Mặc định không cho heuristic fallback nếu model load fail, trừ khi config có `allow_fallback: true`.

Điểm tốt cho benchmark:

> This avoids silently benchmarking a heuristic fallback when model weights or dependencies are missing.

### `core/experiment_engine.py`

Vai trò:

- Trái tim của Stage 1.
- Tổ chức loop benchmark.
- Apply noise.
- Resolve prompt.
- Run prediction.
- Compute metric.
- Save raw CSV and artifacts.
- Resume từ raw CSV đã có.
- Cache noisy images để tránh generate lại.
- Cleanup CUDA/GC giữa các model.

### `models/wrappers/base_model.py`

Vai trò:

- Base interface cho mọi model runner.
- Mọi runner cần có:
  - `load_model()`
  - `predict(image, prompt)`
- Có heuristic fallback, nhưng benchmark chính không nên dùng fallback.

### `models/wrappers/prompt_utils.py`

Vai trò:

- Tạo point/bbox prompt từ GT mask.
- Normalize prompt modes.
- Build kwargs cho SAM/SAM2-compatible predictors.
- Chọn best candidate mask bằng model score.

Đây là file rất quan trọng cho phần methodology.

### `datasets/*`

Vai trò:

- Chuẩn hóa các dataset khác nhau về cùng format:

```python
{
  "image_id": str,
  "image": np.ndarray uint8,
  "mask": np.ndarray binary uint8,
  "meta": dict
}
```

### `noises/*`

Vai trò:

- Định nghĩa các corruption/noise.
- Registry tên noise sang class.
- `NoiseManager` áp dụng noise deterministic theo seed.

### `metrics/metric_manager.py`

Vai trò:

- Tính IoU, Dice, Recall, Precision, F1, Hausdorff Distance.
- Xử lý GT empty bằng `nan`.
- Nếu prediction empty, precision được set 0.0 để tránh hiểu nhầm là perfect.

### `analysis/*`

Vai trò:

- Aggregate raw CSV.
- Merge stats.
- Tính comprehensive statistics.
- Tính robustness analysis như clean score, noisy mean, relative drop, degradation slope, AUC robustness, stability rank.

### `viz/*`

Vai trò:

- Tạo figure/table cho paper.
- Line plots theo severity.
- Heatmap model-noise, model-level, noise-level.
- Ranking model và noise difficulty.
- Prediction overlays.
- Noise galleries.
- Prompt visualization.

## 8. Dataset Design

Dataset adapters hiện có:

1. `ImageMaskFolderAdapter`: generic image/mask folders.
2. `BUSIAdapter`: BUSI ultrasound, mask nằm cùng folder với suffix `_mask`.
3. `CAMUSAdapter`: CAMUS NIfTI cardiac ultrasound.

Trong `configs/full_benchmark.yaml`, datasets gồm:

- Montgomery
- BUSI
- CAMUS NIfTI
- CAMUS 2D folder
- TN3K
- TG3K
- DDTI

Lưu ý quan trọng:

Config hiện có hai dataset entry cùng tên `"CAMUS"`. Nếu muốn phân biệt rõ trong output/paper, nên đổi thành `"CAMUS_NIfTI"` và `"CAMUS_2D"` hoặc chỉ giữ một biến thể. Nếu giữ cùng tên, kết quả có thể bị gộp chung theo cột `dataset`.

## 9. Model Design

Model registry hiện hỗ trợ:

| Paper Name | Runner Key | Wrapper |
|---|---|---|
| SAM | SAM1 | `SAMRunner` |
| SAM2 | SAM2 | `SAM2Runner` |
| SAM3 | SAM3 | `SAM3Runner` |
| MedSAM | MEDSAM1 | `MedSAMRunner` |
| MedicoSAM | MEDICOSAM | `MedicoSAMRunner` |
| MedSAM2 | MEDSAM2 | `MedSAM2Runner` |
| MedSAM3 | MEDSAM3 | `MedSAM3Runner` |
| SAM-Med2D | SAM-MED2D | `SAMMed2DRunner` |
| UltraSAM | ULTRASAM | `UltraSAMRunner` |

Các wrapper dùng chung convention:

```python
runner.load_model()
mask = runner.predict(image, prompt)
```

Điểm thiết kế tốt:

- Model được load một lần cho mỗi model/prompt mode.
- Prediction trả binary mask HxW.
- Wrapper che đi khác biệt API giữa SAM, SAM2, SAM3, MedSAM, UltraSAM.
- ModelManager ngăn benchmark nhầm heuristic fallback.

## 10. Prompt Protocol

Prompt modes chính:

### `prompt_point`

- Dùng foreground point sinh từ GT mask.
- Với object rời rạc, có thể tự chọn representative points theo components.
- Enforce point-only: không truyền bbox.

### `prompt_bbox`

- Dùng bounding box sinh từ GT mask.
- Có adaptive margin.
- Enforce bbox-only: không truyền point.

### `prompt_point_box`

- Dùng một foreground point và một GT-derived box.
- Dùng để đánh giá prompt combined.

### Vì sao dùng GT-derived prompts?

Mục tiêu benchmark là đo robustness của segmentation model dưới noise, không đo chất lượng prompt generator. Dùng GT-derived prompt giúp giảm nhiễu thực nghiệm do prompt sampling.

Câu paper:

> To isolate segmentation robustness from prompt-generation variability, all prompts are deterministically derived from the ground-truth mask following the same protocol across models.

## 11. Candidate Mask Selection

Trong SAM-style models, model có thể trả nhiều candidate masks. Project chọn final mask như sau:

1. Request `multimask_output=True`.
2. Lấy `masks` và `iou_predictions` hoặc `scores`.
3. Gọi `select_best_mask(masks, iou_predictions)`.
4. Nếu score hợp lệ, chọn candidate có score cao nhất.
5. Nếu không có score, fallback chọn candidate đầu tiên.

Điều quan trọng:

- Đây là predicted IoU/confidence score của model.
- Không phải IoU thật với ground truth.
- GT chỉ dùng sau đó để tính metric.

UltraSAM point+box:

- Config có `point_box_fusion: "best_score"`.
- Runner chạy box branch và point branch.
- Chọn mask có prediction score cao hơn.
- Không dùng GT để quyết định point-mask hay box-mask.

MedSAM point mode:

- Có logic riêng với logits threshold.
- Giữ component quanh click.
- Chọn candidate theo area khi cần.
- Vẫn không chọn bằng GT metric.

## 12. Noise Protocol

Noise được apply on-the-fly, không lưu sẵn vào dataset.

Noise types trong full benchmark:

- gaussian
- speckle
- salt_pepper
- motion_blur
- jpeg
- pixelation
- low_brightness
- high_brightness
- low_contrast
- high_contrast
- rician
- poisson

Levels:

- L0: clean
- L1: very mild
- L2: mild
- L3: moderate
- L4: strong
- L5: severe
- L6: extreme
- L7: destructive
- L8: near failure
- L9: catastrophic

Seed design:

```text
effective_seed = hash(base_seed, noise_seed, dataset, image_id, noise_type, level)
```

Điểm mạnh:

- Cùng một ảnh, noise type, level và seed sẽ luôn sinh cùng corruption.
- Khác dataset/image/noise/level sẽ có seed khác.
- Benchmark reproducible.

Lưu ý:

`full_benchmark.yaml` hiện đặt `n_noise_seeds: 1`. Nếu viết paper mạnh hơn về statistical robustness, nên cân nhắc chạy `n_noise_seeds: 3` hoặc hơn.

## 13. Metric Protocol

Metrics:

- IoU
- Dice
- Recall
- Precision
- F1
- HD, tức Hausdorff Distance

Metric direction:

- Higher-is-better: IoU, Dice, Recall, Precision, F1.
- Lower-is-better: HD.

Handling special cases:

- GT empty: metric trả `nan`, vì không có object để evaluate.
- Prediction empty khi GT non-empty:
  - Precision = 0.0.
  - HD = `inf`.
- Aggregation thay `inf` bằng `nan` trước khi tính mean/std.

## 14. Output Structure

Experiment root mặc định:

```text
outputs/full_benchmark/
```

Các output chính:

```text
outputs/full_benchmark/
  raw_files_manifest.csv
  statistics_merged.csv
  stage1b_summary.csv
  noise_cache/
  artifacts/
  statistics/
  visualizations/
```

Raw CSV được ghi theo:

```text
outputs/full_benchmark/<dataset>/<model>/<runner>_<dataset>_<prompt>_raw.csv
```

Mỗi raw row chứa:

- dataset
- model
- prompt_mode
- noise_type
- noise_level
- noise_seed
- image_id
- prompt_x, prompt_y
- bbox coordinates
- gt foreground pixels
- pred foreground pixels
- empty flags
- IoU, Dice, Recall, Precision, F1, HD

Artifacts gồm:

- original image
- noisy image
- GT mask
- predicted mask

## 15. Aggregation and Analysis

Stage 1b dùng:

- `analysis.stats_merger.StatisticsMerger`
- `analysis.aggregator.MetricAggregator`

Group keys:

```text
dataset, model, prompt_mode, noise_type, noise_level
```

Aggregated statistics:

- mean
- std
- coefficient of variation percent
- number of valid metric values
- number of images
- GT empty rate
- prediction empty rate

Comprehensive statistics thêm:

- overall summary
- model summary
- prompt/mode summary
- noise summary
- level summary
- model-noise matrix
- model-level matrix
- noise-level matrix
- robustness analysis

Robustness analysis gồm:

- clean_score
- noisy_mean
- relative_drop_pct
- degradation_slope
- auc_robustness
- stability_rank

## 16. Visualization Outputs

Stage 2 tạo các visualizations dùng cho paper:

- Line plots theo severity L0-L9.
- Heatmaps cho model x noise.
- Heatmaps cho model x level.
- Heatmaps cho noise x level.
- Model ranking.
- Noise difficulty ranking.
- Segmentation gallery.
- Noise gallery.
- Prompt comparison.
- Statistical tables.

Các figure nên chọn cho paper:

1. Main robustness curve: Dice/IoU vs severity.
2. Model x noise heatmap.
3. Noise difficulty ranking.
4. Prompt mode comparison.
5. Qualitative examples: clean, noisy, GT, prediction.

## 17. Reproducibility Features

Project đã có các điểm tốt cho reproducibility:

- Config-driven benchmark.
- Stable output root resolution.
- Deterministic noise seed.
- Resume từ raw CSV.
- Manifest file listing raw files.
- Noisy image cache.
- Artifacts lưu lại examples.
- Model fallback kiểm soát bằng config.
- Multi-GPU split deterministic theo model list.

## 18. Fairness and Methodology Notes

Các điểm cần viết rõ để reviewer không hiểu nhầm:

1. Prompt là GT-derived, tức oracle prompt.
2. Candidate mask selection là non-oracle.
3. Metric được tính sau khi final mask đã chọn.
4. Cùng prompt protocol dùng cho mọi model/prompt mode.
5. Cùng noise protocol dùng cho mọi dataset/model.
6. Fallback heuristic không được dùng trong benchmark chính.
7. Nếu có filter dataset/model khi chạy, paper phải ghi rõ subset.

Suggested wording:

> All models are evaluated under the same deterministic prompt and corruption protocol. Prompts are derived from ground-truth masks to control prompt variability. Candidate mask selection is oracle-free and uses only model-provided confidence or predicted IoU scores. Ground-truth masks are used exclusively for prompt construction and final metric computation, not for selecting among predicted candidates.

## 19. Potential Limitations

Các hạn chế nên chủ động nhắc trong paper:

1. GT-derived prompt không phản ánh hoàn toàn workflow clinical thực tế.
2. Benchmark đo robustness của segmentation given prompt, không đo robustness của automatic prompt generation.
3. Nếu `n_noise_seeds=1`, variance theo random corruption chưa được đo mạnh.
4. Một số model wrapper phụ thuộc external repos/weights, cần đảm bảo đúng checkpoint.
5. Dataset paths là local, cần mô tả dataset preprocessing rõ.
6. Hai entry CAMUS cùng name có thể làm kết quả khó phân biệt.
7. HD có thể không ổn định khi prediction empty hoặc object rất nhỏ.

## 20. Recommended Code Improvements

Các idea nâng cấp code trước khi chốt paper:

### Add Selected Mask Metadata

Nên log thêm:

- selected_mask_index
- selected_mask_score
- mask_selection_policy
- n_candidate_masks

Lợi ích:

- Chứng minh rõ non-oracle selection.
- Dễ debug khi model fail.

### Add Prompt Provenance Columns

Nên log thêm:

- prompt_source = `gt_derived`
- n_points
- prompt_has_bbox
- prompt_area_ratio

Lợi ích:

- Reviewer nhìn raw CSV sẽ thấy protocol rõ.

### Rename Ambiguous Comment

Trong `select_best_mask`, docstring hiện nói "IoU score". Nên đổi thành "model-predicted IoU/confidence score" để tránh bị hiểu là oracle IoU.

### Increase Noise Seeds

Nếu đủ compute:

```yaml
noise_config:
  base_seed: 42
  n_noise_seeds: 3
```

Lợi ích:

- Mean/std đáng tin hơn.
- Có thể report confidence interval.

### Add Confidence Intervals

Trong analysis có thể thêm:

- bootstrap 95% CI
- paired comparison between models
- Wilcoxon/Friedman test theo dataset/noise/prompt

### Add Object-Size Stratification

Có thể chia samples theo `gt_fg_pixels`:

- small lesion/object
- medium
- large

Lợi ích:

- Y tế rất quan trọng vì lesion nhỏ dễ mất dưới noise.

### Add Clean-to-Noisy Relative Robustness Score

Một score đơn giản:

```text
Robustness = mean_noisy_score / clean_score
```

Cho higher-is-better metrics. Với HD cần đảo hoặc dùng relative increase.

### Add Auto-Prompt Baseline

Hiện có prompt từ GT. Có thể thêm baseline:

- center point
- image heuristic prompt
- detector-generated bbox

Nhưng nên để là future work nếu paper tập trung vào robustness given prompt.

## 21. Suggested Paper Structure

### Abstract

- Medical SAM models are sensitive to real-world image degradation.
- Propose unified robustness benchmark.
- Evaluate multiple SAM-family models across datasets, noise types, severity levels, and prompt modes.
- Use GT-derived prompts to control prompt variability.
- Select candidate masks without oracle access.
- Report degradation, rankings, and failure patterns.

### Method

1. Datasets.
2. Models.
3. Prompt protocol.
4. Noise protocol.
5. Inference and mask selection.
6. Metrics.
7. Statistical aggregation.

### Experiments

1. Overall robustness.
2. Model comparison.
3. Noise difficulty.
4. Prompt mode sensitivity.
5. Dataset-specific behavior.
6. Qualitative failure cases.

### Discussion

- Medical-adapted models may or may not be robust depending on noise type.
- Box prompt may be more stable than point under severe corruption.
- Certain noise types cause stronger boundary or object disappearance errors.
- GT-derived prompts isolate segmentation robustness but are not a full interactive clinical setting.

## 22. Methods Paragraph Draft

You can adapt this:

> We evaluate SAM-family models under a unified medical image segmentation robustness protocol. For each dataset sample, controlled image corruptions are applied on-the-fly at severity levels L0-L9 using deterministic seeds. Each model is evaluated under three prompt settings: point, bounding box, and point-plus-box. To isolate segmentation robustness from prompt-generation variability, prompts are deterministically derived from the ground-truth mask using the same protocol across all models. When a model returns multiple candidate masks, the final prediction is selected using the model-provided predicted IoU or confidence score, without using ground-truth masks or evaluation metrics. Ground-truth masks are used only for prompt construction and final metric computation.

## 23. Mask Selection Paragraph Draft

> Candidate mask selection is non-oracle. For SAM-compatible models, we request multiple candidate masks and select the mask with the highest model-predicted IoU/confidence score. For UltraSAM point-box fusion, the point- and box-prompt predictions are fused by selecting the branch with the higher prediction score. No ground-truth mask, Dice, IoU, or hindsight evaluation score is used during candidate selection.

## 24. Experiment Commands

Quick smoke test:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 2 --datasets BUSI --models SAM2 --device cpu
```

Run one dataset and one model on GPU:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --datasets BUSI --models SAM2 --device cuda:0
```

Aggregate after run:

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

Generate visualizations:

```bash
python main.py --config configs/full_benchmark.yaml --stage visualize
```

Full:

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

Multi-GPU:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

## 25. File Map

```text
.
├── main.py                         # CLI and 3-stage orchestration
├── configs/
│   ├── full_benchmark.yaml          # main full benchmark config
│   ├── phase1.yaml
│   └── phase2.yaml
├── core/
│   ├── config_manager.py            # YAML config loading and device resolution
│   ├── model_manager.py             # model runner registry and loading
│   └── experiment_engine.py         # Stage 1 inference loop
├── datasets/
│   ├── dataset_registry.py          # adapter registry
│   └── adapters/
│       ├── image_mask_folder_adapter.py
│       ├── busi_adapter.py
│       └── camus_adapter.py
├── models/
│   └── wrappers/
│       ├── base_model.py
│       ├── prompt_utils.py
│       ├── sam_runner.py
│       ├── sam2_runner.py
│       ├── sam3_runner.py
│       ├── medsam_runner.py
│       ├── medicosam_runner.py
│       ├── medsam2_runner.py
│       ├── medsam3_runner.py
│       ├── sam_med2d_runner.py
│       └── ultrasam_runner.py
├── noises/
│   ├── noise_manager.py             # deterministic on-the-fly noise controller
│   ├── noise_registry.py            # noise registry
│   ├── base.py
│   └── individual noise files
├── metrics/
│   └── metric_manager.py            # IoU, Dice, Recall, Precision, F1, HD
├── analysis/
│   ├── aggregator.py                # raw -> stats
│   ├── stats_merger.py              # merge all stats
│   ├── comprehensive_statistics.py  # robustness summaries
│   └── comprehensive_visualization.py
├── viz/                             # paper figures and visual helpers
├── scripts/
│   └── run_real_mode_matrix.py      # real-inference validation matrix
└── outputs/                         # generated benchmark outputs
```

## 26. What To Emphasize In Presentation

Nên nói project theo flow này:

1. Problem: SAM models need robustness evaluation in noisy medical images.
2. Protocol: same datasets, same prompts, same noise severity, same metrics.
3. Engineering: wrappers unify many SAM variants behind one interface.
4. Fairness: GT-derived prompt controls prompt variation; mask selection is non-oracle.
5. Reproducibility: deterministic noise, config-driven, raw CSV, artifacts.
6. Results: ranking, degradation curves, heatmaps, qualitative failures.
7. Limitations: GT prompt, number of seeds, dataset-specific preprocessing.

## 27. Final Takeaway

Dự án này không chỉ là script chạy SAM. Nó là một benchmark framework có cấu trúc rõ:

- Config decides experiment.
- Dataset adapters normalize data.
- Noise manager controls corruption.
- Prompt utils standardize prompt.
- Model wrappers standardize inference.
- Metric manager standardizes evaluation.
- Analysis and visualization produce paper-ready evidence.

Ý tưởng khoa học chính:

> Evaluate and explain how SAM-family medical segmentation models degrade under controlled image quality shifts, while keeping prompt and mask-selection protocols consistent across models.

