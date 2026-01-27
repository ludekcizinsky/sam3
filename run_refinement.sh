#!/bin/bash
set -euo pipefail

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate sam3
module load gcc ffmpeg

scene_name=$1
prompt_frame=$2
output_dir=/scratch/izar/cizinsky/thesis/preprocessing/$scene_name
mkdir -p $output_dir

cd submodules/sam3
python refinement.py --text "a person" --scene-dir $output_dir --prompt-frame $prompt_frame
