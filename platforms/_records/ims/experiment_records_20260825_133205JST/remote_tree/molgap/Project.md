PROJECT BRIEF — 市售有机电子材料分子的 HOMO/LUMO/Gap 机器学习预测数据库

 给 Claude Code 的项目说明书。所有事实均已实测验证。
 作者背景:TUAT 工学部 化学物理工学科,指导老师布置课题。

0. 一句话目标

构建一个市售有机电子材料分子(OLED / 有机薄膜 / 有机太阳能电池相关)的电子性质数据库,用机器学习从分子结构预测 HOMO energy、LUMO energy、HOMO-LUMO gap 三个量,最终能对「买得到但还没算过」的分子给出性质预测。

1. 对题目的完整理解

1.1 任务本质

市售分子 (OLED / 有机薄膜 / 太阳能电池, 分子量 200–300 起步)
   ↓ 获取 / 计算
HOMO energy, LUMO energy, HOMO-LUMO gap
   ↓ 机器学习 (从分子结构学三个量)
预测模型
   ↓ 应用
对市售分子预测性质 → 建立可查询数据库


1.2 已确认的约束与事实

- 目标性质(3个,多目标回归):HOMO、LUMO、HOMO-LUMO gap
- 分子量范围:先从 200–300 g/mol 小分子开始(Gaussian 算得动)
- 材料领域:OLED、有机薄膜、有机太阳能电池及 building blocks
- 数据源:老师点名 PubChemQC(已验证可用)
- 量子化学工具:Gaussian / GaussView,分子研(IMS)服务器可用
- 数据性质:PubChemQC 的值是计算值(B3LYP/6-31G* level),非实验值
- 流程顺序:先用现成数据库拿数据,缺的再 Gaussian 补算,别一上来跑 Gaussian

1.3 物理意义(写报告用)

- HOMO = 最高被占分子軌道;LUMO = 最低空分子軌道;gap = E(LUMO) − E(HOMO)
- 对 OLED:gap 影响发光颜色/电荷传输/激发能;HOMO 高→空穴传输好,LUMO 低→电子传输好

2. 数据源(已实测验证)

2.1 主数据源:PubChemQC B3LYP/6-31G*//PM6

- HF: molssiai-hub/pubchemqc-b3lyp
- 约 8594 万分子,MW up to 1000
- 几何 PM6 优化,电子性质 B3LYP/6-31G* 单点
- 整库 ≈ 7.67 TB(1587 文件,JSON)→ 不可整库下载

2.2 关键:内部已切好子集

data/b3lyp_pm6_chon300nosalt/train/*.json
  → 元素限 C,H,O,N; 分子量 < 300; 无盐
  → 87 个文件, 共 ≈ 349 GB, 每文件 2–4 GB

其他可选:b3lyp_pm6_chon500nosalt、b3lyp_pm6_chnopsfclnakmgca500

2.3 字段(实测样例 CID=1)

{
  "cid": 1,
  "pubchem-molecular-weight": 203.23558,
  "pubchem-molecular-formula": "C9H17NO4",
  "pubchem-isomeric-smiles": "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
  "energy-alpha-homo": -4.60960862747,
  "energy-alpha-lumo": -0.26122929648,
  "energy-alpha-gap":   4.34837933099,
  "coordinates": [...],         // 大字段, 丢弃
  "orbital-energies": [[...]]   // 大字段, 丢弃
}


单位说明(务必代码核对):homo/lumo 实测负值量级 ~-4到-8,gap 字段 ≈ lumo−homo 数值一致。开工第一步先写单位自检脚本(取几条算 lumo−homo 对比 gap)。若需 eV,Hartree×27.2114。闭壳层分子 alpha==beta,用 energy-alpha-* 即可。

2.4 流式读取(已实测成功)

- HF 文件支持 HTTP Range(返回 206),可只拉前段
- JSON 按 CID 升序排列,文件名即 CID 区间
- ijson 增量解析,只下 12MB 就解析出真实 HOMO/LUMO/gap,不落盘大文件

import urllib.request, ijson, io
url = "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/resolve/main/{file}"
req = urllib.request.Request(url, headers={"User-Agent":"curl/8","Range":"bytes=0-12000000"})
buf = urllib.request.urlopen(req, timeout=60).read()
for obj in ijson.items(io.BytesIO(buf), "item"):
    cid=obj.get("cid"); mw=obj.get("pubchem-molecular-weight")
    homo=obj.get("energy-alpha-homo"); lumo=obj.get("energy-alpha-lumo"); gap=obj.get("energy-alpha-gap")
    # 边读边过滤 MW, 只留需要字段


2.5 市售信息补充源

TCI(東京化成)、Sigma-Aldrich/Merck、Ossila、Lumtec、Alfa Aesar、PubChem(拿 CID/SMILES/式/MW)

3. 全流程

Phase 1 — 数据获取(流式,不落盘大文件)
1.1  单位自检:抓第一文件前~20条,确认 gap==lumo-homo
1.2  过滤条件:MW∈[200,300], 元素⊆{C,H,O,N}
1.3  流式扫 chon300nosalt 87文件:Range分块 + ijson增量解析 + 边读边过滤
     只留 cid/MW/formula/SMILES/homo/lumo/gap,丢 coordinates、orbital-energies
1.4  产出 data/raw/pubchemqc_chon_mw200_300.csv(预计几万~十几万分子,几十MB)

 磁盘有限,只存精简 CSV,绝不落盘原始 JSON

Phase 2 — 清洗与特征
2.1  去重(canonical SMILES/InChI)
2.2  异常值过滤(gap<0、离群、解析失败)
2.3  单位统一(建议建模用 eV)
2.4  RDKit:Morgan fingerprint(ECFP4,r=2,2048bit)+ 描述符
     (MW,MolLogP,TPSA,NumAromaticRings,HBD,HBA,RotatableBonds,FractionCSP3,杂原子数)
2.5  划分:建议 scaffold split(Bemis-Murcko),否则 random 8:1:1


Phase 3 — 建模(多目标回归)
3.1  baseline:LightGBM/XGBoost/RandomForest,输入=fingerprint(+描述符),输出=[HOMO,LUMO,gap]
3.2  指标:对 HOMO/LUMO/gap 分别报 MAE,RMSE,R²
3.3  进阶(可选):GNN(PyG 的 MPNN/GIN/SchNet)
3.4  误差分析:哪类骨架预测差、gap误差分布


Phase 4 — 预测与数据库
4.1  收集市售分子清单(TCI/Sigma/Ossila + CID)
4.2  无 PubChemQC 数据的市售分子用模型预测
4.3  关键分子 Gaussian B3LYP/6-31G(d) 补算验证
4.4  汇总数据库:分子名|CID|SMILES|式|MW|用途|供应商|HOMO|LUMO|gap|来源|备注


4. 推荐目录结构 (1/2)
oled_homo_lumo_ml/
├── PROJECT_BRIEF.md
├── requirements.txt
├── src/  01_fetch_stream.py / 02_clean.py / 03_features.py / 04_train.py / 05_predict.py / utils.py
├── data/  raw/ processed/ commercial/
├── models/  notebooks/  results/


5. 技术栈
Python 3.10+
获取:urllib/requests(Range), ijson
处理:pandas, numpy
化学:rdkit
ML:lightgbm/xgboost/scikit-learn  (进阶可选 torch, torch_geometric)
可视化:matplotlib, seaborn


6. 开工顺序(给 Claude Code)
1. 不要整下 7.67TB;不要第一步跑 Gaussian
2. 先写 01_fetch_stream.py,只抓 chon300nosalt 第一文件前 200 条,跑通解析+过滤+写CSV
3. 写单位自检:确认 gap≈lumo-homo,确认 Hartree/eV
4. 跑通后扩到全子集流式过滤 MW 200–300
5. 03_features.py RDKit fingerprint+描述符
6. 04_train.py LightGBM baseline,输出 MAE/RMSE/R²
7. 看效果再决定上 GNN / 补 Gaussian
8. 小步验证,每脚本独立可跑、有日志


7. 报告必写点
- HOMO/LUMO/gap 全是计算值(B3LYP/6-31G* level),非实验值
- 几何 PM6 优化,电子性质 B3LYP 单点
- 原始 Hartree,展示用 eV(注明换算)
- ML 预测值 vs PubChemQC 计算值要明确区分来源
- 数据划分方式(random/scaffold)影响 R²,要说明

8. 开工前问老师 1 个问题
建模用 fingerprint+描述符(LightGBM)还是分子图 GNN? 不确认也行,先 baseline 跑通拿结果再问。
