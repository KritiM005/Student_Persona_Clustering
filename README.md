# Student Persona Discovery using PCA, UMAP, t-SNE and K-Means Clustering

## Overview

This project explores how students differ in their academic behavior, AI adoption patterns, career preparedness, learning habits, and psychological characteristics. Using unsupervised machine learning techniques, students are grouped into meaningful personas that capture distinct behavioral and career-development profiles.

The analysis combines dimensionality reduction, cluster validation, visualization techniques, and K-Means clustering to uncover hidden structures within the data.

## Problem Statement

Students differ significantly in how they use AI tools, prepare for careers, develop skills, manage stress, and approach learning. Traditional demographic segmentation often fails to capture these multidimensional behavioral patterns.

The objective of this project is to identify naturally occurring student personas using clustering techniques and provide interpretable profiles that can support educational and career-related decision-making.

## Dataset

- Source: Kaggle
- Records: 15,000 students
- Features: 30 variables
- Data Type: Numerical and categorical

The dataset includes information on:

- Academic background
- AI tool usage
- Career readiness
- Placement preparation
- Learning habits
- Skill development
- Stress and burnout indicators
- Motivation and self-development

## Methodology

### Data Preparation

- Identifier removal
- Categorical encoding
- Feature standardization

### Dimensionality Reduction

- Principal Component Analysis (PCA)

### Cluster Selection

Multiple clustering configurations were evaluated using:

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

### Clustering

- Algorithm: K-Means
- Final Clusters: 4
- Clustering Space: PCA-transformed features

### Cluster Visualization

To evaluate cluster structure and separability, multiple visualization approaches were used:

- PCA Projection
- t-SNE Projection
- UMAP Projection

### Persona Profiling

Cluster-level feature comparisons were used to interpret and label the discovered student personas.

## Final Model Performance

| Metric | Value |
|----------|----------|
| Silhouette Score | 0.1335 |
| Davies-Bouldin Index | 1.8560 |

A nested clustering approach was also investigated. However, the flat K-Means solution achieved superior validation performance and was selected as the final segmentation model.

### Statistical Validation

To verify that the discovered personas differ significantly on key behavioral and career-related attributes:

- One-Way ANOVA
- Tukey's HSD Post-Hoc Analysis
- 
## Key Outcomes

- Identified four distinct student personas.
- Revealed meaningful differences in AI adoption, career preparedness, and learning behavior.
- Demonstrated the usefulness of combining PCA, UMAP, and t-SNE for cluster exploration.
- Established an interpretable segmentation framework suitable for educational and career-focused applications.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PCA
- K-Means
- t-SNE
- UMAP

## Repository Structure

```text id="2mn6ev"
Student-Persona-Clustering/
│
├── Student_Persona_Clustering.ipynb
├── README.md
├── docs/
│   └── Results_and_Insights.md
└── results/
    ├── model_selection_plots.png
    ├── pca_visualization.png
    ├── tsne_visualization.png
    ├── umap_visualization.png
    └── cluster_profiles.png
```

## Detailed Analysis

A detailed discussion of preprocessing, dimensionality reduction, cluster validation, visualization results, persona interpretation, limitations, and future improvements is available in:

**docs/Results_and_Insights.md**

## Author

**Kriti Maheshwari**  
M.Sc. Statistics, IIT Kanpur
