# ColorS2O: Controllable SAR-to-Optical Image Translation with Sparse Colour Anchors

Official code for the paper of the same title (under review). If you use it, please cite the paper (see [Citation](#citation)).

ColorS2O translates a SAR image into an RGB optical image. A sparse set of colour anchors controls the colours of the output. Anchors are small dots whose colours come from an optical observation of the same area. This repository contains the translator, for training and inference, and the anchor-sampling tool used in the paper.

The code calls colour anchors *colour hints*. An anchor map is an image in `color_hint_by_dots/` that is black wherever there is no anchor.

## Repository contents

| Path | Purpose |
| --- | --- |
| `main_jit.py` | Entry point for training and evaluation |
| `engine_jit.py` | Training loop, epoch-level loss schedule and generation |
| `denoiser.py` | Diffusion wrapper, colour losses and their time-dependent weights |
| `model_jit_mask_guided_embed.py` | ColorS2O backbone with Mask-Guided Sparse Embedding (MGSE) and gated colour cross-attention |
| `model_jit.py` | Backbone without MGSE, used for the ablation |
| `hint_mask_generator_SAR2Opt_percentage.py` | Anchor sampling with SAM 2 |
| `util/` | Data loading, learning-rate schedule and distributed helpers |
| `sam2/` | SAM 2 inference code (Apache License 2.0) |
| `fid_stats/` | ImageNet FID statistics from JiT, used by the built-in FID/IS report |
| `FLOPs.py` | FLOPs and parameter count of the backbone |

## Installation

```bash
conda env create -f environment.yaml
conda activate jit
```

`environment.yaml` pins Python 3.10, PyTorch 2.5.1 and CUDA 12.4, and installs all other dependencies. Anchor generation, training and evaluation run on CUDA GPUs. The perceptual loss uses LPIPS with a VGG-16 backbone, and torchvision downloads the VGG-16 weights on first use.

Download the SAM 2.1 Hiera-L checkpoint:

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The config name `configs/sam2.1/sam2.1_hiera_l.yaml` resolves inside the bundled `sam2/` package.

## Data layout

```text
<DATA_ROOT>/
├── trainA/   # training SAR images
├── trainB/   # training optical images: supervision and training anchors
├── testA/    # test SAR images
└── testB/    # test optical images: reference and test anchors
```

The loader pairs files in sorted order and stops with an error if two paired names differ. Names must match exactly, including the extension. Anchor maps are always written as `<name>.png`, so store the SAR and optical images as PNG files.

## 1. Generate colour anchors

Run the generator on the optical images of each split:

```bash
python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/trainB" \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --devices 0 \
  --output-dir "<TRAIN_ANCHOR_DIR>" \
  --summary-csv "<TRAIN_ANCHOR_DIR>/summary.csv"

python hint_mask_generator_SAR2Opt_percentage.py \
  --input "<DATA_ROOT>/testB" \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --devices 0 \
  --output-dir "<TEST_ANCHOR_DIR>" \
  --summary-csv "<TEST_ANCHOR_DIR>/summary.csv"
```

Each output directory contains:

```text
<ANCHOR_DIR>/
├── color_hint/          # original optical pixels under each dot, black elsewhere
├── color_hint_by_dots/  # each dot filled with its dominant colour: the anchor map
├── hint_masks/          # dot masks per region (.npz)
├── label_map/           # SAM 2 region labels (16-bit PNG)
├── label_overlay/       # visualisation of the regions
└── summary.csv          # realised anchor coverage per image (hint_mask_ratio)
```

- Pass `<ANCHOR_DIR>/color_hint_by_dots` as `--hint_train_path` or `--hint_test_path`. The `color_hint` folder is an alternative that keeps the per-pixel colours under each dot.
- `--devices` lists the GPUs to use, one worker per GPU, for example `0` (the default) or `0,1,2,3`.
- Images that already have an output in `--output-dir` are skipped, so an interrupted run can be restarted. Use a new output directory after changing the sampling settings.

### Sampling settings

The defaults in `hint_defaults` and `seg_defaults`, inside `generate_hint_masks()`, are the SAR2Opt settings. For the 256 × 256 SEN1-2 images we used smaller dots, a smaller spacing and different SAM 2 filtering thresholds. `hint_defaults` overrides the keyword defaults of `build_hint_masks_from_label_map()`.

| Paper | Code | SAR2Opt | SEN1-2 |
| --- | --- | --- | --- |
| Nominal coverage target ρ | `target_ratio` | `0.05` | `0.05` |
| Colour-dominance threshold τ_dom | `histogram_threshold` | `0.6` | `0.6` |
| Minimum spacing d_min (px) | `min_point_distance` | `10` | `8` |
| Dot radius (px) | `dot_radius_range` | `(1.0, 6.0)` | `(1.0, 3.0)` |
| Colour quantisation | `histogram_bins` | `16` levels per RGB channel | `16` |
| Random seed | `rng_seed` | `34` | `33` |
| SAM 2 mask score threshold | `score_thr` | `0.3` | `0.45` |
| SAM 2 minimum region area (px) | `min_area` | `500` | `300` |
| SAM 2 maximum number of masks | `max_keep` | `3000` | `800` |

A candidate dot is kept only if it lies entirely inside one SAM 2 region and passes the dominance check on its own pixels. The spacing is enforced between dot centres in the same region. All accepted dots in one region are filled with a single colour: the dominant colour of their union, quantised to 16 levels per channel.

The realised coverage in `summary.csv` can be lower than the target, because candidates that fail these checks are rejected. To change the coverage, edit `target_ratio`, for example to `0.01` or `0.03`. To disable the dominance check for the ablation, set `histogram_threshold=0`. For a fair comparison, change only that value and keep the data, the other settings and the training commands fixed.

### Ablation without MGSE

`denoiser.py` imports the full model:

```python
from model_jit_mask_guided_embed import JiT_models
```

For the variant without MGSE, replace this line with:

```python
from model_jit import JiT_models
```

This variant embeds the anchor map and its mask with a plain patch embedding. The gated colour cross-attention is unchanged. Keep everything else fixed.

## 2. Training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29501 main_jit.py \
  --output_dir "<OUTPUT_DIR>" \
  --sar_train_path "<DATA_ROOT>/trainA" \
  --opt_train_path "<DATA_ROOT>/trainB" \
  --hint_train_path "<TRAIN_ANCHOR_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --hint_loss_weight 0 \
  --enabled_losses ab perc
```

- `--enabled_losses ab perc` is the configuration used in the paper. It adds the CIELAB chromatic loss (`ab`) and the LPIPS perceptual loss (`perc`) to the velocity loss. The code default, `ab perc sam ssim`, also enables a spectral-angle loss and an SSIM loss, so always pass this option.
- The curriculum defaults are the values used in the paper. The chromatic loss uses `--lambda_ab_max 0.75 --lambda_ab_t0 0.75 --lambda_ab_alpha 10`, and the perceptual loss uses `--lambda_perc_max 0.3 --lambda_perc_t0 0.55 --lambda_perc_alpha 5`. The epoch-level ramp is `--epoch_schedule cosine`.
- `--hint_loss_weight 0` keeps the velocity loss unweighted at anchor pixels.
- `--batch_size` is the batch size per GPU (default 2). The learning rate is `--blr` × total batch size / 256.
- Checkpoints are written to `<OUTPUT_DIR>/checkpoint-last.pth` every `--save_last_freq` epochs (default 5), and to `checkpoint-<epoch>.pth` every 100 epochs.
- To continue a run, add `--resume "<OUTPUT_DIR>"`. The option takes a directory and loads `checkpoint-last.pth` from it. If that file does not exist, training starts from scratch and the log shows a warning instead of `Resumed checkpoint from`.
- `--sync_cuda_every_step ''` skips a CUDA synchronisation after every step that is only needed for debugging.

### Loss ablations

Change only `--enabled_losses`:

```bash
--enabled_losses perc      # perceptual loss only
--enabled_losses ab        # chromatic loss only
--enabled_losses ab perc   # both, as in the paper
```

### Loss activation order

Each colour loss is activated along the diffusion time by its midpoint `t0`:

```bash
--lambda_ab_t0 0.75     # chromatic loss, default
--lambda_perc_t0 0.55   # perceptual loss, default
```

A smaller `t0` makes the loss active over a wider and noisier range of diffusion times. With the defaults, the perceptual loss is activated before the chromatic loss. To reverse the order, use, for example, `--lambda_ab_t0 0.50 --lambda_perc_t0 0.75`.

## 3. Testing and inference

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29502 main_jit.py \
  --resume "<OUTPUT_DIR>" \
  --output_dir "<EVAL_DIR>" \
  --sar_test_path "<DATA_ROOT>/testA" \
  --opt_test_path "<DATA_ROOT>/testB" \
  --hint_test_path "<TEST_ANCHOR_DIR>/color_hint_by_dots" \
  --img_size 512 \
  --evaluate_gen True \
  --gen_bsz 8 \
  --keep_outputs True \
  --use_hint_infer True \
  --enabled_losses ab perc
```

- Generation uses the EMA weights of the checkpoint and the Heun sampler with 50 steps (`--sampling_method heun --num_sampling_steps 50`).
- Images are written to `<EVAL_DIR>/heun-steps50-cfg1.0-interval0.0-1.0-image50000-res512/` under the SAR file names.
- `--resume` loads `<OUTPUT_DIR>/checkpoint-last.pth`. To evaluate another checkpoint, copy it as `checkpoint-last.pth` into a directory of its own. If no checkpoint is found, the script prints a warning and evaluates a randomly initialised model.
- Use the same `--enabled_losses` as for training, because the LPIPS network is stored in the checkpoint.
- `--evaluate_gen`, `--use_hint_infer`, `--keep_outputs` and `--sync_cuda_every_step` take a string. Any non-empty value enables the option, including `False`. To disable an option, pass an empty string, for example `--use_hint_infer ''`.
- With `--use_hint_infer ''`, the same weights translate from SAR alone. No anchor input reaches the colour branch, and the optical and anchor folders are not read.
- After generation, the script prints FID and Inception Score against the ImageNet statistics in `fid_stats/`. These are not the metrics reported in the paper, which are computed on the saved images. This step supports `--img_size` 256 and 512 only, and `--keep_outputs ''` deletes the generated images after it.

## Trained models

Trained weights are available from the corresponding author on reasonable request.

## Licence

The JiT-derived code is released under the MIT License (`LICENSE`). The `sam2/` directory is under the Apache License 2.0 (`sam2/LICENSE`). See `THIRD_PARTY_NOTICES.md`.

## Acknowledgements

ColorS2O builds on [JiT](https://github.com/LTH14/JiT) and uses [SAM 2](https://github.com/facebookresearch/sam2), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [Kornia](https://github.com/kornia/kornia) and [torch-fidelity](https://github.com/toshas/torch-fidelity).

## Citation

If you use this code, please cite the paper. GitHub's "Cite this repository" button uses `CITATION.cff`; the journal reference will be added after publication.

```bibtex
@misc{he2026colors2o,
  title        = {ColorS2O: Controllable SAR-to-Optical Image Translation with Sparse Colour Anchors},
  author       = {He, Jingfei and Shi, Hao and Chen, Yuhang and Fan, Rui and Gamba, Paolo and Chen, Liang},
  year         = {2026},
  note         = {Manuscript under review},
  howpublished = {\url{https://github.com/Saturn-He/ColorS2O}}
}
```
