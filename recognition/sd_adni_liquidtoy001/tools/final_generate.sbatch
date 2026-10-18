#!/bin/bash
#SBATCH --job-name=final-gen
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_generate_%j.out
#SBATCH --error=logs/final_generate_%j.err
# Stage 1 of docs/final-evaluation-protocol.md: one model's cohort, seeds 37200
# onwards, COUNT per class (32 in the rehearsal, 512 in the final run).
# Usage: sbatch tools/final_generate.sbatch MODEL COUNT
#   MODEL: sd e0 e2 r1e2 e3 r1e3 b3 b1 b2 b2b
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
name="${1:?model required}"
count="${2:?seeds per class required}"
out="outputs/final-$name-$SLURM_JOB_ID"
python=.venv-train/bin/python
case "$name" in
    b1) $python tools/sample_vae.py --checkpoint "$B1_CHECKPOINT" --output "$out" --count "$count" ;;
    b2) $python tools/baseline_gan.py generate --checkpoint "$B2_CHECKPOINT" --output "$out" --count "$count" ;;
    b2b) $python tools/baseline_gan.py generate --class-checkpoint "AD=$B2B_AD" \
        --class-checkpoint "NC=$B2B_NC" --output "$out" --count "$count" ;;
    *)
        if [ "$name" = e0 ] && [ ! -f "${WEIGHTS[e0]}" ]; then
            $python tools/export_e0_adapter.py --checkpoint "$E0_CHECKPOINT" --output "${WEIGHTS[e0]}"
        fi
        diffusion_args "$name"
        $python predict.py "${model_args[@]}" --output "$out" --count "$count" --batch-size 16
        ;;
esac
