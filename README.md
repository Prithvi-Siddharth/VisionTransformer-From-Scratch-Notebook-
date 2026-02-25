# Vision Transformer from Scratch – CIFAR-10 Classification

Implementation of the **Vision Transformer (ViT)** paper ("An Image is Worth 16×16 Words") **from scratch** in PyTorch — trained on **CIFAR-10**.

No external pre-trained models, no high-level libraries like timm or transformers — pure PyTorch + understanding.

## What this repository contains

- Pure from-scratch implementation of **Vision Transformer**
- Patch embedding + learnable positional embeddings + class token
- Multi-head self-attention with residuals & layer normalization
- MLP (feed-forward) inside each transformer layer
- Training loop on **CIFAR-10** (32×32 images)
- Simple but clean model definition
- Visualization of random predictions

## Model Architecture Summary

1. **Input**: RGB image (3 × 32 × 32)
2. **Patch Embedding**  
   - Split image into fixed-size patches (e.g. 4×4 or 8×8)  
   - Linear projection → flattened patch embeddings
3. **Class Token** + **Positional Embeddings**  
   - Prepend learnable `[CLS]` token  
   - Add learnable 1D positional encodings
4. **Transformer Encoder** (repeated N times)  
   - LayerNorm → Multi-Head Self-Attention → residual  
   - LayerNorm → MLP (feed-forward) → residual
5. **Classification Head**  
   - Take the `[CLS]` token output after final layer  
   - Linear layer → 10 logits (CIFAR-10 classes)

## Features

- Device-agnostic (CPU / CUDA)
- Standard training + validation loop
- Learning rate ~3e-4, AdamW / Adam optimizer
- Batch size 128–256 (depending on GPU memory)
- Dropout and layer normalization used as in original ViT
- No heavy data augmentation in base version (you can easily add CutMix, MixUp, AutoAugment, etc.)


## Repository Structure
