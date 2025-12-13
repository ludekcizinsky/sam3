#!/bin/bash
set -euo pipefail

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate sam3
module load gcc ffmpeg

scane_name=$1
prompt_frame=$2
output_dir=/scratch/izar/cizinsky/thesis/results/$scane_name

cd submodules/sam3
python inference.py --text "a person" --output-dir $output_dir --prompt-frame $prompt_frame