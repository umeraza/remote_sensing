# Evaluation metrics

- **AAB grounding:** Pr@0.5, Pr@0.7, Pr@0.9, mIoU, cIoU.
- **RAB grounding:** rotated AP50, AP75, and rotated mAP.
- **Mask grounding:** mask AP, AP50, AP75, oIoU, and mIoU.
- **Counting:** MAE, RMSE, and exact count accuracy.
- **Language tasks:** VQA accuracy, scene accuracy, CIDEr, BLEU-4, METEOR, ROUGE-L.
- **Spatial consistency:** Spatial Agreement Score (SAS), computed after converting RAB and mask outputs to enclosing AABs.
- **Expert utilization:** average expert load, routing entropy, and expert activation frequency.
