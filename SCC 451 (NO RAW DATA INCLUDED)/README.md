# SCC451 Machine Learning Coursework

Comprehensive machine learning project demonstrating exploratory data analysis, preprocessing, clustering, and deep learning classification.

## Project Overview

This coursework consists of two main components:

# 2. Climate Data Analysis** (Basel Climate Dataset)
   - Exploratory Data Analysis (EDA)
   - Data Preprocessing Pipeline
   - Unsupervised Clustering Analysis

# 1. Image Classification** (Oxford Pets Dataset)
   - Binary Classification: Cat vs Dog
   - Custom CNN Architecture
   - Deep Learning with PyTorch

# Project Structure

```
ML_Coursework/
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── ClimateDataBasel.csv
│   │   └── images/
│   │       ├── Cat/
│   │       └── Dog/
│   └── processed/            # Preprocessed data
│       └── climate_data_processed.csv
│
├── src/                      # Source code modules
│   ├── utils.py              # Shared utility functions
│   ├── preprocessing/        # Data preprocessing
│   │   ├── data_exploration.py
│   │   └── data_preprocessing.py
│   ├── clustering/           # Clustering analysis
│   │   └── climate_clustering.py
│   └── classification/       # Image classification
│       └── pet_classifier.py
│
├── scripts/                  # Execution scripts
│   ├── run_eda.py
│   ├── run_preprocessing.py
│   ├── run_clustering.py
│   ├── run_classification.py
│   └── generate_summary_report.py
│
├── results/                  # Generated outputs
│   ├── figures/              # Visualizations (PNG)
│   ├── tables/               # Results tables (CSV)
│   └── models/               # Trained models (PyTorch)
│
└── README.md                 # This file
```

# Quick Start

# 1. Setup Environment

```bash
# Ensure you have Python 3.8+ installed
pip install -r requirements.txt
```

# 2. Run Complete Pipeline

Execute the following scripts in order:

```bash
# Step 1: Exploratory Data Analysis
python scripts/run_eda.py

# Step 2: Data Preprocessing
python scripts/run_preprocessing.py

# Step 3: Clustering Analysis
python scripts/run_clustering.py

# Step 4: Image Classification (requires image dataset)
python scripts/run_classification.py

# Step 5: Generate Summary Report
python scripts/generate_summary_report.py
```

# 3. View Results

- Figures: `results/figures/` (13+ PNG visualizations)
- Tables: `results/tables/` (8+ CSV files)
- Model: `results/models/pet_classifier_best.pth`
- Summary Report: `results/PROJECT_SUMMARY.md`

# Key Features

# Exploratory Data Analysis
- Statistical summary and distribution analysis
- Missing value detection
- Outlier detection (IQR and Z-score methods)
- Correlation analysis with heatmap
- Feature variance analysis

# Data Preprocessing
- Correlation-based feature selection (r > 0.95)
- Outlier handling with IQR method
- Feature scaling with StandardScaler
- Complete preprocessing pipeline

# Clustering Analysis
- Elbow method for optimal K determination
- Silhouette analysis for cluster validation
- K-means clustering
- Hierarchical clustering (Ward linkage)
- Dendrogram visualization
- Cluster characterization and profiling

# Image Classification
- Custom CNN architecture (4 conv blocks + 3 FC layers)
- Data augmentation (RandomCrop, RandomHorizontalFlip)
- Training with early stopping
- Comprehensive evaluation metrics
- Confusion matrix and sample predictions

# Results Summary

# Classification Performance
- Test Accuracy: 95.13%
- Precision: 94.60%
- Recall: 95.73%
- F1-Score: 95.16%

# Clustering Quality
- Optimal Clusters: K = 2
- Silhouette Score: ~0.32 (good separation)
- Method: K-means recommended

# Dataset Statistics
- Climate Data: 1,762 samples × 18 features
- Image Data: 49,996 images (balanced: 50% Cat, 50% Dog)

Technologies Used

# Core Libraries
- Python 3.13
- PyTorch 2.7.1 (CUDA 11.8)
- scikit-learn 1.5.2
- pandas 2.2.3
- numpy 2.1.3

# Visualization
- matplotlib 3.9.2
- seaborn 0.13.2

# Hardware
- GPU: NVIDIA GeForce RTX 3060 Ti
- CUDA Version: 11.8

# Outputs Generated

# Figures
1. EDA visualizations (5)
2. Clustering analysis (6)
3. Classification results (3)

# Tables
1. Statistical summaries
2. Correlation matrices
3. Outlier analysis
4. Cluster profiles
5. Classification metrics

# Models
- Trained CNN: `pet_classifier_best.pth` (26M parameters)

# Course References

All implementations reference appropriate course materials:
- **Lecture 2:** Data preprocessing and understanding
- **Lab 2:** Matplotlib histograms
- **Lab 3:** Statistical analysis and correlation
- **Lab 4:** K-means, hierarchical clustering, validation metrics
- **Lab 5:** Image data loading and augmentation
- **Lab 6:** CNN architecture, training, and evaluation

#Reproducibility

To reproduce results:
1. Use the same random seed (42)
2. Follow the execution order above
3. Ensure identical package versions
4. Use the same data splits (70/15/15)

All configurations, hyperparameters, and model weights are saved for reproducibility.
