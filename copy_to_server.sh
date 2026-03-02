#!/bin/bash

# ── Configuration ────────────────────────────────────────────────────────────
SSH_CONFIG="bioinf02"   # used as: ssh bioinf02
SERVER_PATH="~/Desktop/2025-Master-Thesis/thesis_code"
LOCAL_PATH="."           # run this script from your project root
DATE=$(+%Y%m%d%H%M%S)
# ── Generate requirements.txt ────────────────────────────────────────────────
echo "Generating requirements.txt with pipreqs..."
mv requirements.txt requirements.txt.bak.${DATE}
pipreqs --force "$LOCAL_PATH" \
    --ignore .venv,notebooks,data,logs,docs,plots,results
echo "Done: requirements.txt"

# ── Sync to server ───────────────────────────────────────────────────────────
echo "Syncing to $SSH_CONFIG:$SERVER_PATH ..."
rsync -avz --progress -e "ssh" \
    --exclude='.venv/' \
    --exclude='notebooks/' \
    --exclude='data/00_external_data' \
    --exclude='data/00_problems_data' \
    --exclude='data/01_raw' \
    --exclude='data/02_cleaned' \
    --exclude='data/04_models' \
    --exclude='data/05_model_output' \
    --exclude='logs/' \
    --exclude='docs/' \
    --exclude='plots/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='*.egg-info/' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='README' \
    --exclude='general_knowledge.md' \
    --exclude='LSTM-inner-window.jpg' \
    --exclude='LSTM_inner-environment*' \
    # ── Add more exclusions below as needed ──────────────────────────────
    # --exclude='some_other_dir/' \
    "$LOCAL_PATH/" "$SSH_CONFIG:$SERVER_PATH"

echo "Done. Files synced to server."

# ── Optional: recreate conda env on server ───────────────────────────────────
# Uncomment if you want to also trigger env setup remotely via SSH:
# echo "Setting up conda environment on server..."
# ssh "$SSH_CONFIG" << 'EOF'
#     conda env create -f ~/thesis_code/environment.yml || conda env update -f ~/thesis_code/environment.yml
#     conda activate thesis_ls
#     pip install -r ~/thesis_code/requirements.txt
# EOF
# echo "Remote environment ready."