#  Multimodal Regression for E-Commerce Price Estimation

> **A complete stacking-based machine learning system to predict log-transformed product prices using text, images, and tabular data.**
>
> The project demonstrates how multimodal signals outperform single-modality approaches when predicting messy, high-variance real-world pricing.

---

## Motivation

E-commerce pricing is not a simple function.  
A product’s value is influenced by:

- Visual features (quality, material, color, brand indicators)
- Textual signals (feature descriptions, sentiment, specs, keywords)
- Structured metadata (ratings, quantity, category, attributes)

Traditional models (pure tabular, pure text, pure image) collapse in cases like:
- Same product / different brands
- Apparel vs electronics
- High-variance categories with outliers

This project solves that by **fusing modality embeddings** and then using a **2-Level Stacking Ensemble** to learn stable price predictors.

---

#  Overview

This repository contains a full pipeline that:

1. Extracts embeddings from product text + images
2. Fuses them with engineered tabular features
3. Trains 4 independent base regressors
4. Feeds their Out-of-Fold predictions into a meta-learner
5. Generates final predictions

The system is architected to reduce model variance and improve real-world pricing generalization.

---

# Core Design Philosophy

> **Don’t fight the modalities — leverage them.**
>
> Boosting models dominate tabular tasks, while neural networks dominate unstructured tasks.  
> The key is to **let each model do what it’s best at**, and then aggregate them.

---

# Dataset Modalities

The system expects three input categories:

### **1️Text → Transformer Embeddings**
- Product title
- Description
- Attribute text
- Additional unstructured signals

### ** Images → SigLIP Embeddings**
- Product photos
- Logos / brand imprints
- Visual patterns

### **3️Tabular Metadata**
Examples:
- Category
- Rating
- Stock
- Sales rank
- Quantity per unit (IPQ)
- Derived metrics (value_per_item, discount ratio)

---

#  System Architecture    

       ┌──────────────┐
       │ Raw Dataset  │
       └──────┬───────┘
              |
   ┌──────────────────────────────┐
   │        Feature Pipeline      │
   └──────────────────────────────┘
   Text Emb. │ Image Emb. │ Tabular
      │          │          │
   PCA(128)   PCA(128)   Engineered
       └──────┬──────┬──────┘
             Concatenate
                  │
          Multimodal Vector
                  │
       ┌──────────────────┐
       │  Base Models     │
       └──────────────────┘
 LGBM │ XGB │ CatBoost │ MLP
                  │
         OOF Predictions
                  │
          Meta Learner (LGBM)
                  │
            Final Price


