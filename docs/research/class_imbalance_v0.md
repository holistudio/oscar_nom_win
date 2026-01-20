# Handling class imbalance in long-document transformer classification

**Binary classification on movie screenplays with severe imbalance (4-5% to 19% positive class), extremely long inputs (37k tokens average), and limited GPU memory requires a hierarchical chunking architecture with document-level loss weighting—not traditional resampling methods.** For your specific constraints (n=2000, RTX 4090 with 16GB VRAM), the evidence strongly favors combining class-weighted focal loss with a ChunkBERT-style hierarchical transformer. SMOTE and text-level oversampling fail for long documents and should be avoided entirely. Recent 2024 research on contrastive learning methods (CAROL, SharpReCL) shows promise for imbalanced text classification [1](https://discovery.researcher.life/article/class-aware-contrastive-optimization-for-imbalanced-text-classification/a95bb629360b3c55a0207a9cecf11474), while LLM-based data augmentation offers the most effective way to expand minority class samples.

## Why 30k+ tokens fundamentally changes the approach

Standard transformers max out at **512-4,096 tokens**—far below your 37,000-token average. Even efficient attention models like Longformer (4,096 tokens) [2](https://arxiv.org/pdf/2201.11838) and LED (16,384 tokens) [3](https://github.com/allenai/longformer) cannot process your documents in a single pass on 16GB VRAM. The math is unforgiving: attention matrices for 30k tokens would require approximately **7.2GB just for one attention head** at fp16 precision, before accounting for model weights, gradients, or optimizer states.

This constraint eliminates most off-the-shelf approaches and forces a **hierarchical architecture** where documents are split into chunks, processed individually, then aggregated for classification. The good news: empirical research shows ChunkBERT-style approaches **outperform Longformer by 5.7% on average** across long document benchmarks while using substantially less memory.

The critical interaction with class imbalance emerges here: when you split documents into chunks, a 100k-token screenplay creates ~200 chunks while a 7k-token screenplay creates only ~14. If you compute loss per-chunk, longer documents dominate training regardless of class distribution. **Always apply class weights at the document level after aggregation**, using document counts (not chunk counts) to calculate weights.

## Class imbalance techniques ranked by effectiveness

For your **4-5% positive class** (~80-100 samples in training), certain techniques dramatically outperform others:

**Highly effective approaches:**
- **Class-weighted focal loss** (γ=2.0, α based on inverse frequency) reduces contribution from easy negatives while upweighting minority class samples. Multiple 2023-2024 studies validate this for transformer-based text classification with severe imbalance.
- **Threshold tuning** is essential—the optimal decision threshold for 4-5% positive class is typically **0.01-0.05**, not the default 0.5 [4](https://pmc.ncbi.nlm.nih.gov/articles/PMC7769929/). Use ROC analysis or precision-recall curves on validation data to find optimal thresholds.
- **Effective number of samples weighting** (from CVPR 2019) outperforms simple inverse frequency for extreme imbalance: `weights = (1 - β) / (1 - β^n)` where β=0.9999 and n is class count.

**Moderately effective approaches:**
- **LLM-based data augmentation** can generate 50-200 synthetic minority class screenplays. A 2024 MDPI study found generating 10-20 new samples per label via GPT-4 consistently improves minority class performance. This is far more effective than SMOTE for text data.
- **Weighted random sampling** ensures minority class appears in training batches, though with batch size 1-2 (required for your memory constraints), this matters less since weights are applied per-sample regardless.

**Approaches to avoid:**
- **SMOTE on text or embeddings** destroys semantic meaning—interpolating between two 37k-token documents produces vectors that don't represent valid text [5](https://www.blog.trainindata.com/overcoming-class-imbalance-with-smote/). The ACL 2023 survey explicitly states SMOTE "is hardly applicable to textual data."
- **Random undersampling** is catastrophic for n=2000: reducing ~1,900 negative samples to ~80-100 throws away the vast majority of your limited training data.
- **Chunk-level class weighting** artificially inflates or deflates class balance based on document length rather than true class distribution.

For your **19% positive class** (~380 samples), the problem is less severe. Simple inverse frequency weighting (class_weight=[1.0, 4.26]) combined with threshold tuning typically suffices.

## Memory-efficient architecture for 16GB VRAM

The optimal architecture for your constraints combines hierarchical processing with aggressive memory optimization:

**Document processing pipeline:**
```
Screenplay (37,000 tokens)
    ↓ Split with overlap
72 chunks × 512 tokens (50-token overlap)
    ↓ Process in mini-batches of 8
Chunk embeddings (72 × 768 dimensions)
    ↓ Transformer aggregation layer
Document embedding (1 × 768)
    ↓ Classification head with weighted loss
Binary prediction
```

**Essential memory optimizations** (all compatible with class imbalance techniques):
- Mixed precision training (bf16/fp16): **50% activation memory savings**
- Gradient checkpointing: **~75% activation memory savings** at cost of 20-33% slower training
- 8-bit AdamW optimizer: **50% optimizer state savings**
- Flash Attention: **O(N²) → O(N) memory** for attention computation

With these optimizations, a BERT-base encoder with hierarchical aggregation fits comfortably within 16GB: ~500MB model weights, ~1GB optimizer states, ~2-3GB activations with checkpointing, leaving headroom for chunk processing.

**Concrete configuration:**
```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch 16
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_bnb_8bit",
)
```

Batch size 1 may seem concerning, but recent research (arXiv 2507.07101) demonstrates it performs comparably to larger batches when Adam's β₂ is scaled appropriately. Class weights function correctly at batch size 1 because they weight individual sample contributions, not batch-level statistics.

## Recent advances worth implementing (2023-2025)

Three research directions from 2024 offer meaningful improvements over baseline approaches:

**CAROL (Class-Aware Contrastive Optimization)** from October 2024 combines reconstruction loss with contrastive class separation, specifically addressing the problem that standard methods produce embeddings with class overlap for minority classes. It outperformed BERT+Focal Loss, RoBERTa, and GPT-3.5/4 few-shot approaches on imbalanced text datasets.

**SharpReCL** uses prototype vectors for each class and ensures every class appears at least once per batch through mixup-based hard example generation. Tested successfully on RTX 3090Ti (24GB) [6](https://arxiv.org/html/2405.11524), it should adapt to your hardware with reduced batch sizes.

**LLM-AutoDA** (NeurIPS 2024) uses LLMs to automatically search for optimal augmentation strategies for long-tailed distributions. While computationally expensive during the search phase, the resulting augmented datasets significantly improve downstream classifier performance.

For your specific combination of extreme document length + severe imbalance + small dataset, **no directly comparable published research exists**. This gap means your results could contribute novel findings to the field.

## Implementation roadmap for Oscar prediction

Given your specific task (predicting Oscar nominations/wins from screenplays), here's the recommended approach ordered by priority:

**Phase 1: Baseline architecture**
- Implement ChunkBERT pattern with RoBERTa-base encoder (512-token chunks, 64-token overlap)
- Two-layer transformer aggregator over chunk embeddings
- Document-level focal loss (γ=2.0) with effective number of samples weighting
- Train with gradient checkpointing, bf16, batch size 1, gradient accumulation 16

**Phase 2: Address the 4-5% win prediction separately**
The Oscar win prediction (4-5% positive, ~66 wins in training) is substantially harder than nomination prediction. Consider:
- Train a nomination classifier first, then use nomination probability as a feature for win prediction
- Generate 50-100 synthetic winning screenplays via GPT-4 prompt engineering
- Use more aggressive focal loss (γ=3.0) or asymmetric loss functions
- Implement threshold tuning—expect optimal threshold around 0.02-0.04

**Phase 3: Enhance with contrastive learning**
If baseline performance is insufficient, implement CAROL-style contrastive loss to improve minority class embedding separation. This adds complexity but addresses the fundamental problem of overlapping class representations.

## Evaluation and monitoring considerations

Standard accuracy is meaningless for 4-5% positive class (95.5% accuracy by predicting all negative) [7](https://pmc.ncbi.nlm.nih.gov/articles/PMC7769929/).  Use these metrics instead:

- **Macro F1 score**: Balances precision and recall across classes equally
- **Precision-Recall AUC**: More informative than ROC-AUC for severe imbalance
- **Per-class precision/recall**: Track minority class performance explicitly
- **Calibration plots**: Class weighting can distort probability estimates; calibrate post-training if you need well-calibrated probabilities

For validation, use **stratified k-fold cross-validation** (k=5) to ensure each fold preserves class distribution. With only ~66 wins in training data, each fold will have ~13 positive examples—monitor for high variance across folds.

## Conclusion

Your problem sits at an intersection of three challenging constraints: severe class imbalance, extremely long sequences, and limited training data. The research evidence points clearly toward a hierarchical architecture (ChunkBERT pattern) with document-level focal loss weighting as the most practical and effective approach for your 16GB VRAM constraint.

**The single most important insight**: class imbalance handling must occur at the document level after chunk aggregation—not at the chunk level or on raw text. SMOTE and traditional oversampling techniques fail entirely for long-text transformers and should be avoided.

For the 4-5% Oscar win prediction specifically, expect this to remain challenging even with optimal techniques. Consider whether the business problem truly requires predicting wins versus nominations, or whether nomination prediction alone provides sufficient value. The nomination task (19% positive) is substantially more tractable and may deliver most of the practical utility.