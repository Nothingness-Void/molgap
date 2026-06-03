"""
MolGap — Colab 嵌入提取脚本
============================
在 Google Colab (免费 T4 GPU) 上运行。
提取 ChemBERTa 和 MolFormer 两个模型的 SMILES 嵌入向量。

使用方法：
  1. 打开 Google Colab (colab.research.google.com)
  2. 菜单 → Runtime → Change runtime type → T4 GPU
  3. 上传 pubchemqc_chon_mw200_300_clean.csv
  4. 新建代码单元格，粘贴本脚本全部内容，运行
  5. 运行完毕后下载两个嵌入 CSV 文件
"""

# ============================================================
# Cell 1: 安装依赖
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers==4.40.2", "tokenizers<0.20",
                       "huggingface_hub<0.25", "torch", "pandas", "tqdm"])

# ============================================================
# Cell 2: 导入 & 检查 GPU
# ============================================================
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Cell 3: 上传数据
# ============================================================
# 方法 A：左侧文件面板手动上传（推荐）
#   点击 Colab 左侧 📁 图标 → 点击上传按钮 → 选择 CSV 文件
#   上传完成后再运行下面的代码
#
# 方法 B：从 Google Drive 挂载
#   先把 CSV 上传到 Google Drive，然后取消注释下面两行：
# from google.colab import drive
# drive.mount('/content/drive')
# INPUT_FILE = "/content/drive/MyDrive/pubchemqc_chon_mw200_300_clean.csv"

INPUT_FILE = "pubchemqc_chon_mw200_300_clean.csv"

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"找不到 {INPUT_FILE}\n"
        "请先通过左侧文件面板上传，或从 Google Drive 挂载。"
    )

df = pd.read_csv(INPUT_FILE)
smiles_list = df["canonical_smiles"].tolist()
cid_list = df["cid"].tolist()
print(f"Loaded {len(smiles_list)} molecules")

# ============================================================
# Cell 4: 通用嵌入提取函数
# ============================================================
def extract_embeddings(model_name, smiles, cids, batch_size=64, max_length=128):
    """
    加载 HuggingFace 模型，对 SMILES 做 mean-pooling 提取嵌入。
    返回 DataFrame: cid + emb_0, emb_1, ...
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # 检查嵌入维度
    dummy = tokenizer("C", return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        dummy_out = model(**dummy)
    hidden_dim = dummy_out.last_hidden_state.shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    all_embeddings = []
    n_batches = (len(smiles) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Extracting"):
        batch = smiles[i * batch_size : (i + 1) * batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            output = model(**tokens)

        # Mean pooling over non-padding tokens
        hidden = output.last_hidden_state          # (B, seq_len, dim)
        mask = tokens["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # 简短前缀
    short_name = model_name.split("/")[-1].lower().replace("-", "_")[:12]
    col_names = [f"{short_name}_{j}" for j in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    emb_df.insert(0, "cid", cids)
    return emb_df

# ============================================================
# Cell 5: 提取 ChemBERTa 嵌入
# ============================================================
chemberta_df = extract_embeddings(
    model_name="seyonec/ChemBERTa-zinc-base-v1",
    smiles=smiles_list,
    cids=cid_list,
    batch_size=64,
)
chemberta_out = "embeddings_chemberta.csv"
chemberta_df.to_csv(chemberta_out, index=False)
print(f"Saved: {chemberta_out} ({chemberta_df.shape})")

# ============================================================
# Cell 6: 提取 MolFormer 嵌入
# ============================================================
molformer_df = extract_embeddings(
    model_name="ibm/MoLFormer-XL-both-10pct",
    smiles=smiles_list,
    cids=cid_list,
    batch_size=32,  # MolFormer 更大，batch 小一点
    max_length=202,  # MolFormer 推荐 max_length
)
molformer_out = "embeddings_molformer.csv"
molformer_df.to_csv(molformer_out, index=False)
print(f"Saved: {molformer_out} ({molformer_df.shape})")

# ============================================================
# Cell 7: 合并嵌入 + 下载
# ============================================================
merged = chemberta_df.merge(molformer_df, on="cid", how="inner")
merged_out = "embeddings_all.csv"
merged.to_csv(merged_out, index=False)
print(f"\nMerged embeddings: {merged.shape}")
print(f"  ChemBERTa dims: {chemberta_df.shape[1] - 1}")
print(f"  MolFormer dims: {molformer_df.shape[1] - 1}")
print(f"  Total dims: {merged.shape[1] - 1}")
print(f"Saved: {merged_out}")

# 自动触发下载
try:
    from google.colab import files as colab_files
    print("\n正在下载文件...")
    colab_files.download(chemberta_out)
    colab_files.download(molformer_out)
    colab_files.download(merged_out)
except ImportError:
    print(f"\n非 Colab 环境，请手动复制以下文件:")
    print(f"  {chemberta_out}")
    print(f"  {molformer_out}")
    print(f"  {merged_out}")

print("\n完成！将下载的 CSV 放到 data/processed/ 目录下。")
