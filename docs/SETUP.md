# Data and runtime setup

## Local paths and dependencies

Run the commands below from the RRF root. Create the environment using the installation commands in the README, then configure local paths:

```bash
cp paths.env.example paths.env
# Edit paths.env to match your local paths.
source paths.env
```

| Variable | Local content |
| --- | --- |
| `PHYSTWIN_DATA` | Processed scene data and appearance assets |
| `PHYSTWIN_ROOT` | Optional compatible Gaussian rendering backend |
| `PGND_ROOT` | Compatible data-preparation, evaluation and rendering backend |

RRF's particle and regional networks, response feedback, rollout training and model inference are implemented in this package. Dataset adapters consume the processed formats described below. PGND preparation/evaluation and Gaussian rendering also load backend components from the configured paths; these paths must point to an existing compatible installation. PhysTwin geometry-only training/evaluation uses the processed data directly.

When using a separate rendering environment, make the RRF package available there:

```bash
python -m pip install -e /absolute/path/to/RRF --no-deps
python -m pip install -r requirements-render.txt
```

Use a rendering environment with compatible PyTorch, CUDA and rasterization extensions.

## PhysTwin data

`PHYSTWIN_DATA` is the root containing `data/different_types/`. Additional clothing sequences use `additional_data/data/different_types/`. Each processed scene needs:

| File | Content |
| --- | --- |
| `final_data.pkl` | Object positions, visibility/motion-validity masks, and control-point trajectories |
| `split.json` | Official train/test frame intervals and sequence length |
| `gt_track_3d.pkl` | Independent manual tracks, required for the Track metric |

Preserve original scene names and frame indices. Missing manual tracks leave the corresponding metric unavailable. Rendering additionally requires the released RGB images, masks, calibrated cameras, and trained Gaussian appearance files under the official dataset layout.

Prepare the example scenes:

```bash
python -m rrf.prepare phystwin \
  --data-root "$PHYSTWIN_DATA" --out prepared/phystwin \
  --groups double_stretch_zebra single_lift_rope
```

Use `--groups all` to prepare the main scenes and category groups defined in `rrf/configs/phystwin_groups.json`. Reuse already prepared directories; preparation does not overwrite them.

## PGND data

RRF accepts native state caches containing `episode_XXXX/0000.pt`, subsequent frame files, and `info.yaml`. Raw subepisode directories and their metadata are also needed for episode/frame alignment.

Copy `rrf/configs/pgnd_paths.example.json` to `pgnd_paths.json` and set the raw-data and state-cache locations for each category. For example:

```json
{
  "rope": {
    "raw_source": "/absolute/path/to/rope_merged/sub_episodes_v",
    "native_state": "/absolute/path/to/rope_native/state"
  }
}
```

Prepare the category in the configured preprocessing environment:

```bash
python -m rrf.prepare pgnd \
  --pgnd-root "$PGND_ROOT" --paths pgnd_paths.json \
  --out prepared/pgnd --tasks rope
```

Use `--tasks all` after configuring all six categories: `rope`, `cloth`, `sloth`, `box`, `paperbag`, and `bread`. To generate native caches from raw subepisodes, omit `native_state` and add `--preprocess`.

The adapter uses the benchmark episode splits, excluding known cloth training/test duplicates. Prepared manifests record the actual partitions. Keep prepared files and native data at their configured paths.

## Custom training and test data

Specify source and target PhysTwin scenes directly:

```bash
python -m rrf.prepare phystwin \
  --data-root "$PHYSTWIN_DATA" --out prepared/custom \
  --group-name cloth_transfer --generalization scene \
  --train-scenes double_lift_cloth_1 double_lift_cloth_3 \
  --test-scenes single_lift_cloth_4
```

Use `--generalization action` for action-transfer groups. Scene groups use complete source sequences; action groups use the observed source intervals. Target scenes are excluded from training and normalization. Multiple groups can be specified with `--split-file`; see `rrf/configs/phystwin_custom.example.json`. Omit custom splits to use the supplied benchmark groups.

For PGND, optional `train_ids` and `test_ids` in each `pgnd_paths.json` entry define custom episode splits. Both lists must be nonempty and disjoint. When training and testing use separate prepared directories, `rrf.pipeline` also accepts `--eval-data` and `--eval-partition`.

## Simulation and rendering

RRF evaluation rolls out the trained dynamics under prescribed control trajectories. The following adapters render RRF predictions with the initial Gaussian appearance assets. Run them in the rendering CUDA environment, with RRF installed and `paths.env` loaded.

### PhysTwin

Generate the rollout and resolve its appearance assets:

```bash
python -m rrf.evaluate \
  --dataset phystwin --data prepared/phystwin/double_stretch_zebra \
  --run runs/zebra --out runs/zebra/eval \
  --prepare-render --device cuda:0

CUDA_VISIBLE_DEVICES=0 python -m rrf.rendering.image_worker \
  --official "$PHYSTWIN_ROOT" --dataset "$PHYSTWIN_DATA" \
  --evaluation runs/zebra/eval/double_stretch_zebra \
  --case double_stretch_zebra \
  --out runs/zebra/eval/double_stretch_zebra/images --save-all

python -m rrf.collect \
  --inputs runs/zebra/eval/summary.json --out runs/zebra/tables
```


### PGND

After evaluating `runs/pgnd_rope`, resolve and render one held-out episode:

```bash
python -m rrf.rendering.pgnd_assets \
  --data prepared/pgnd/rope --episode 651 \
  --out runs/pgnd_rope/assets/episode_0651.json

CUDA_VISIBLE_DEVICES=0 python -m rrf.rendering.pgnd_visual \
  --pgnd-root "$PGND_ROOT" \
  --config prepared/pgnd/rope/pgnd_config.json \
  --trajectory runs/pgnd_rope/eval/episode_0651/trajectories_world.npz \
  --assets runs/pgnd_rope/assets/episode_0651.json \
  --out runs/pgnd_rope/visual/episode_0651
```

The asset resolver uses the original subepisode metadata to locate RGB images, masks, cameras, and the initial Gaussian model. Repeat for the remaining held-out episodes. To aggregate visual results, create a JSON mapping categories to their geometry and visual output directories:

```json
{"rope": {"geometry": "runs/pgnd_rope/eval", "visual": "runs/pgnd_rope/visual"}}
```

```bash
python scripts/collect_pgnd_visual.py \
  --runs visual_runs.json --out runs/visual_tables
```

Save the mapping above as `visual_runs.json`; add other categories as needed. Rendered frames and per-frame scores remain in their episode directories.

### Appearance and perceptual weights

Rendering requires the official appearance assets, camera calibration, and object masks. PGND also uses the gripper/table splat files from its release. Install the pretrained AlexNet/LPIPS weights before offline evaluation:

```bash
python -c "from torchvision.models import alexnet, AlexNet_Weights; alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)"
python -c "import lpips; lpips.LPIPS(net='alex')"
```

The perceptual-metric backend may use a separate pretrained-weight cache. Image metrics are computed only when the required assets and renderer are available.
