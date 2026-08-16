# Student AI Dependency & Career Anxiety — Persona Clustering

Unsupervised clustering project that segments students into behavioral personas based on AI tool reliance, career readiness, and psychological wellbeing indicators.

## About the Data

-**Kaggle Dataset**
- **File:** `ai_dependency_career_anxiety_students.csv`
- **Size:** 15,000 rows, 30 columns, no duplicate rows
- **Unit of observation:** one student per row
- **Feature groups:**
  - Demographics: age, gender, degree type, stream, year of study, college tier, urban/rural
  - AI usage: daily AI tool usage hours, primary AI tool used, frequency of AI use for assignments, AI dependency score, AI-replaces-own-thinking score
  - Career readiness: placement anxiety, fear of job loss to AI, career clarity, internship experience, job applications, resume confidence, interview anxiety, overall career readiness score
  - Study & wellbeing: daily study hours, self-learning hours, skill courses taken, social media hours, sleep hours, stress, burnout, motivation, career counseling seeking
- **Missing values:**
  - `primary_ai_tools_used` (21%) — structural, not random: fully explained by students who reported "Never" using AI for assignments. Filled with a `"None"` sentinel category rather than imputed.
  - `self_learning_hours_per_week`, `social_media_hrs_per_day`, `sleep_hours`, `seeks_career_counseling` (~1.5% each) — random-looking, imputed via KNN / mode.

## Objective

Identify distinct, interpretable student personas based on AI reliance and career/psychological attributes, without using demographic fields to form the clusters (demographics are used only afterward, to describe each persona).

## Pipeline

1. **EDA** — distributions, missingness pattern, correlation check
2. **Missing value handling** — structural sentinel fill + KNN imputation
3. **Preprocessing** — ordinal encoding of `uses_ai_for_assignments`, two engineered composite indices (`ai_reliance_index`, `wellbeing_index`), standard scaling
4. **Redundancy removal** — dropped raw features that were folded into the composite indices, to avoid double-weighting them in distance calculations
5. **Cluster tendency check** — Hopkins statistic (~0.67), confirming real but moderate (not sharply separated) structure
6. **Dimensionality reduction** — PCA retaining 9 components (~83% variance) prior to clustering
7. **Model comparison** — KMeans (elbow, silhouette, Davies-Bouldin, Calinski-Harabasz), Gaussian Mixture Models (BIC/AIC), hierarchical/agglomerative clustering (dendrogram on a stratified sample), and a nested two-stage KMeans variant — compared against a flat KMeans solution
8. **Final model selection** — flat KMeans, k=4, chosen after nested clustering scored lower on silhouette (0.1146 vs. 0.1335) and showed no distinguishable sub-structure on t-SNE/UMAP
9. **Visualization** — t-SNE and UMAP projections (used for visualization only, not for clustering)
10. **Cluster profiling** — standardized heatmap of cluster means + demographic composition, used to name and describe the four personas

## Key Notes on Method

- Demographic fields were deliberately excluded from the clustering feature set and used only to profile clusters afterward, so personas reflect behavior/attitude rather than demographic grouping.
- t-SNE and UMAP were used strictly for visualizing already-formed clusters, never as clustering input, since they distort distances in ways unsuitable for forming clusters.
- Cross-validation in the supervised sense doesn't apply to unsupervised clustering; model selection relied on silhouette, Davies-Bouldin, Calinski-Harabasz, BIC/AIC, and cross-algorithm agreement instead.

## Results Summary

- Final model: KMeans, k=4, on PCA-reduced features
- Silhouette score: 0.1335 | Davies-Bouldin: 1.8560
- Four personas identified — see `persona_report` for details

## Repository Contents

- `ai_dependency_career_anxiety_students.csv` — raw dataset
- `Student_Persona_Clustering.ipynb` — full analysis notebook (Google Colab)
- Persona wrap-up report — final results and cluster descriptions

## Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `umap-learn`

## Caveat

Silhouette and Davies-Bouldin scores indicate real but moderate cluster separation (consistent with the Hopkins statistic of ~0.67) — this is expected for psychographic survey data. Personas should be read as overlapping tendencies, not hard-walled segments.
