# GPTrans-T precision comparison on Kaggle1

This desktop-owned experiment compares the same GPTrans-T seed-42 100K recipe
under FP32 and CUDA FP16 autocast with FP32 weights/optimizer/EMA. The frozen
question, measurements, and decision rule are in [protocol.md](protocol.md).
Live remote state belongs in [STATUS.md](STATUS.md). Each precision arm has its
own prospective RML trajectory and terminal evidence.
