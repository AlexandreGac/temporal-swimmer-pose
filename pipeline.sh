#!/bin/bash
#SBATCH --job-name=swimmer_data_gen
#SBATCH --partition=gpu             # Replace with your cluster GPU partition name
#SBATCH --account=YOUR_ACCOUNT      # Replace with your SLURM project/account ID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 1 GPU requested
#SBATCH --cpus-per-task=8           # Adjust based on cluster quota
#SBATCH --mem=64G                   # Adjust based on cluster quota
#SBATCH --time=08:00:00
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

set -e  # Exit immediately if a command exits with a non-zero status

# Path to Blender executable (defaults to system 'blender' if not set in environment)
BLENDER_BIN="${BLENDER_PATH:-blender}"

# 1. Prepare directory structure
echo "==> Creating output and log directories..."
mkdir -p results/render
mkdir -p results/synth
mkdir -p results/validation
mkdir -p logs

# 2. Environment activation (uncomment/adapt based on your cluster setup)
# module purge
# module load python/3.12
# source venv/bin/activate

# 3. Blender Rendering & Keypoint Extraction
echo "==> Running Blender synthetic rendering pipeline..."
srun "$BLENDER_BIN" -b background_final_scene.blend -y -P render_pipeline.py

# 4. Annotation Validation & Dataset Packing
echo "==> Running annotation checks..."
srun python check_annotations.py

echo "==> Packing multi-frame dataset into pickle format..."
srun python pack_to_pickle.py

echo "==> Pipeline completed successfully!"