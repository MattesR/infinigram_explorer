#!/bin/bash
# BLEnD answer sampling (CPU only). Submit from the repo root:
#     sbatch blend_sampling.slurm
#
# One job, no follow-ups. Progress is saved after every answer, so if the job hits the
# wall-time limit or is cancelled, submit it again and it continues where it stopped.
# Run only one at a time: the infini-gram API allows ~1 call / 1.3 s per client and
# parallel jobs get the cluster blocked (HTTP 403).
#
#SBATCH --job-name=blend-sampling
#SBATCH --partition=cores
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=blend_sampling-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-.}"

# Python env with requests + transformers (adjust to your setup).
source .venv/bin/activate

# Done already?
if python -u blend_sampling.py status; then
    echo "All regions done."
    exit 0
fi

# The compute node needs outbound HTTPS to the API.
if ! curl -sf -m 20 -o /dev/null -X POST https://api.infini-gram.io/ -H 'Content-Type: application/json' \
        -d '{"index":"v4_olmo-2-0325-32b-instruct_llama","query_type":"count","query":"Algeria"}'; then
    echo "No access to api.infini-gram.io from $(hostname)." >&2
    exit 1
fi

# Locks left by a killed job are stale (only one job runs at a time).
rm -f blend_results/answer_samples/*.lock

python -u blend_sampling.py run --regions "${REGIONS:-all}"
