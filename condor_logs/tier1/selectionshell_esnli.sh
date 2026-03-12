#!/bin/sh
# =============================================================================
# ESNLI Full Pipeline — VSC Tier 1 (Dodrio)
#
# Submits one cache-building job then one training job per model.
# All training jobs depend on the cache job completing successfully.
#
# Usage (from project root):
#   module swap cluster/dodrio/gpu_rome_a100_80
#   bash condor_logs/tier1/selectionshell_esnli.sh
#
# Optional: submit only a subset by commenting out unwanted qsub lines.
# =============================================================================

module swap cluster/dodrio/gpu_rome_a100_80

PBS_CACHE="./condor_logs/tier1/jobfile_esnli_cache.pbs"
PBS_TRAIN="./condor_logs/tier1/jobfile_esnli_train.pbs"
DEFAULT_CFG="./configs/ESNLI/default_config_esnli_tier1.json"
TRAIN_VARS="default_config=${DEFAULT_CFG},fold=0,lr=0.0001,wd=0.0001,batch_size=8"

mkdir -p ./condor_logs/logs_vsc

# -----------------------------------------------------------------------------
# STEP 1: build the v2 cache (train + validation + test)
# Capture the job ID so training jobs can depend on it.
# -----------------------------------------------------------------------------
# CACHE_JOBID=$(qsub "${PBS_CACHE}")
# echo "Cache job submitted: ${CACHE_JOBID}"

# DEPEND="-W depend=afterok:${CACHE_JOBID}"

# -----------------------------------------------------------------------------
# STEP 2: submit one training job per model, each waiting for the cache job
# -----------------------------------------------------------------------------

qsub ${DEPEND} "${PBS_TRAIN}" \
    -v "${TRAIN_VARS},config=./configs/ESNLI/full_lora.json"
echo "Submitted: combined (full_lora)"

qsub ${DEPEND} "${PBS_TRAIN}" \
    -v "${TRAIN_VARS},config=./configs/ESNLI/full_image_lora.json"
echo "Submitted: image-only LoRA (full_image_lora)"

qsub ${DEPEND} "${PBS_TRAIN}" \
    -v "${TRAIN_VARS},config=./configs/ESNLI/full_text_lora.json"
echo "Submitted: text-only LoRA (full_text_lora)"

qsub ${DEPEND} "${PBS_TRAIN}" \
    -v "${TRAIN_VARS},config=./configs/ESNLI/full_image_frozen.json"
echo "Submitted: image-only frozen (full_image_frozen)"

# qsub ${DEPEND} "${PBS_TRAIN}" \
#     -v "${TRAIN_VARS},config=./configs/ESNLI/full_mcr.json"
# echo "Submitted: MCR (full_mcr)"

# qsub ${DEPEND} "${PBS_TRAIN}" \
#     -v "${TRAIN_VARS},config=./configs/ESNLI/full_mmpareto.json"
# echo "Submitted: MMPareto (full_mmpareto)"

# qsub ${DEPEND} "${PBS_TRAIN}" \
#     -v "${TRAIN_VARS},config=./configs/ESNLI/full_dnr.json"
# echo "Submitted: DnR (full_dnr)"

# qsub ${DEPEND} "${PBS_TRAIN}" \
#     -v "${TRAIN_VARS},config=./configs/ESNLI/full_reconboost.json"
# echo "Submitted: ReconBoost (full_reconboost)"

# qsub ${DEPEND} "${PBS_TRAIN}" \
#     -v "${TRAIN_VARS},config=./configs/ESNLI/full_synib.json"
# echo "Submitted: RMask (full_rmask)"

echo ""
echo "All jobs queued. Cache job: ${CACHE_JOBID}"
echo "Training jobs will start automatically once ${CACHE_JOBID} succeeds."
echo "Monitor with: qstat -u \$USER"
