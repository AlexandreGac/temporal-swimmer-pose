# Temporal 2D Swimmer Pose Estimation with Anatomical Constraints and Synthetic Data

Official repository for the 3D procedural synthetic data generation pipeline presented in:

> **Temporal 2D Swimmer Pose Estimation with Anatomical Constraints and Synthetic Data**  
> Alexandre Gachenot, Vincent Lepetit, Rémi Carmigniani

This repository provides the Blender simulation environment, procedural rendering scripts with domain randomization, annotation verification tools, and temporal data packaging routines used to train our Sim2Real underwater pose estimation models.

---

## Repository Structure

```text
├── background_final_scene.blend   # 3D Blender scene (16 rigged meshes, cameras, shaders)
├── render_pipeline.py             # Headless rendering and 2D keypoint projection script
├── check_annotations.py           # Visual verification tool for projected 2D keypoints
├── pack_to_pickle.py              # Preprocessing script (7-frame temporal stacks / 21 channels)
├── pipeline.sh                    # SLURM execution script for GPU clusters
├── results/                       # Generated outputs
├── hdri/                          # HDRI environment maps (CC0 1.0)
├── LICENSE                        # MIT License (source code)
├── ASSETS_LICENSE.md              # License & provenance of bundled assets (CC0)
└── README.md
```

---

## Requirements

* **Blender** >= 5.0 (with Cycles & NVIDIA OptiX / CUDA enabled)
* **Python** >= 3.12
* Python packages (for post-processing & validation):
```bash
pip install numpy opencv-python
```

---

## Usage

### 1. Standalone / Local Execution

You can run the full pipeline step by step:

```bash
# 1. Procedural generation and 2D keypoint extraction in Blender (headless mode)
blender -b background_final_scene.blend -y -P render_pipeline.py

# 2. (Optional) Visually inspect projected keypoints
python check_annotations.py

# 3. Pack temporal sequences into training pickles
python pack_to_pickle.py
```

### 2. SLURM Cluster Execution

To launch the generation on an HPC cluster via SLURM:

1. Edit `pipeline.sh` to specify your cluster account/partition.
2. Submit the job:
```bash
sbatch pipeline.sh
```

---

## Dataset Specifications

The pipeline generates synthetic sequences covering all 4 competitive strokes (freestyle, backstroke, breaststroke, butterfly) alongside starts and flip turns:

* **Domain Randomizations:**
  * 6 Fitzpatrick skin types.
  * 8 contrastive swimwear and cap color palettes.
  * Camera perspective jitter (+-1.0 deg rotation noise).
  * Distractor background swimmers (50% probability).
* **Keypoints (C_kp = 14):** Head, Sternum, Left/Right Shoulders, Elbows, Hands (Wrists), Hips, Knees, Feet (Ankles).
* **Packed Tensor Format (`synth_*.p`):**
  * `images`: Stride-sampled temporal tensor of shape `(H, W, 21)` (7 frames x 3 RGB channels representing temporal window [t-3, t+3]).
  * `body_markers`: Ground-truth coordinate array of shape `(14, 2)` for the central frame t.

---

## 📄 License

* **Source code** (scripts, pipeline): [MIT License](LICENSE).
* **HDRI environment maps** (`hdri/`): [CC0 1.0 Universal](ASSETS_LICENSE.md).
