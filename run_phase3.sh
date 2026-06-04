#!/usr/bin/env bash
set -euo pipefail

# ── MolGap Phase 3 一键运行脚本 ──
# 用法:
#   screen -S molgap
#   bash run_phase3.sh
#   # Ctrl+A D 断开, screen -r molgap 重连
#
# 环境: Linux CPU, Python 3.10+
# 可调参数 (环境变量):
#   LGBM_TRIALS=80   LightGBM Optuna trials (default 80)
#   XGB_TRIALS=60    XGBoost Optuna trials (default 60)
#   MAX_RECORDS=30000 数据获取量 (default 30000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

echo "=========================================="
echo "  MolGap Phase 3 Pipeline"
echo "  $(date)"
echo "=========================================="

# ── 1. 创建虚拟环境 & 安装依赖 ──
if [ ! -f "$PYTHON" ]; then
    echo ">>> [1/5] Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

echo ">>> [1/5] Installing dependencies..."
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt xgboost catboost

# ── 2. 数据获取 (如果 Phase 3 raw 数据不存在) ──
RAW_PATTERN="data/raw/phase3_chonsfcl_mw200_500_*.csv"
PHASE3_FEAT="results/phase3/phase3_features.csv"
MAX_RECORDS="${MAX_RECORDS:-30000}"

if ! ls $RAW_PATTERN 1>/dev/null 2>&1; then
    echo ">>> [2/5] Fetching CHONSFCl MW 200-500 data (${MAX_RECORDS} records)..."
    "$PYTHON" scripts/phase3/scaleup.py --max-records "$MAX_RECORDS" 2>&1 | tail -20
else
    echo ">>> [2/5] Raw data exists, skipping fetch."
fi

# ── 3. Phase 3 baseline (scaleup 含 clean + features + train) ──
if [ ! -f "$PHASE3_FEAT" ]; then
    echo ">>> [3/5] Running Phase 3 scaleup (clean + features + baseline)..."
    "$PYTHON" scripts/phase3/scaleup.py 2>&1 | tail -20
else
    echo ">>> [3/5] Phase 3 features exist, skipping baseline."
fi

# ── 4. Phase 3.4 优化 ──
LGBM_TRIALS="${LGBM_TRIALS:-80}"
XGB_TRIALS="${XGB_TRIALS:-60}"

echo ""
echo "=========================================="
echo "  [4/5] Phase 3.4: Optuna Optimization"
echo "  LightGBM trials: $LGBM_TRIALS"
echo "  XGBoost trials:  $XGB_TRIALS"
echo "=========================================="
echo ""

mkdir -p results/phase3/optimize

"$PYTHON" scripts/phase3/select_and_optimize.py \
    --lgbm-trials "$LGBM_TRIALS" \
    --xgb-trials "$XGB_TRIALS" \
    2>&1 | tee results/phase3/optimize/optimize_log.txt

# ── 5. 完成 ──
echo ""
echo "=========================================="
echo "  [5/5] Done!"
echo "  $(date)"
echo "=========================================="
echo "  Results: results/phase3/optimize/"
echo "  Key files:"
echo "    model_comparison.csv    — 全模型对比"
echo "    optimize_summary.json   — 最佳模型 & 参数"
echo "    best_params_lgbm.json   — LightGBM 最佳超参"
echo "    best_params_xgb.json    — XGBoost 最佳超参"
echo "=========================================="
