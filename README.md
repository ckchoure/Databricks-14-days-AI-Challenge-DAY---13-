# Day 13 – Model Comparison & Feature Engineering  
Databricks 14 Days AI Challenge

## Objective
The goal of Day 13 was to build a fresh machine learning workflow from scratch,
train multiple models on the same dataset, compare their performance using MLflow,
and select the best-performing model using a Spark ML Pipeline.

---

## Tasks Covered
1. Train 3 different machine learning models  
2. Compare model metrics using MLflow  
3. Build Spark ML Pipelines  
4. Select the best-performing model  

---

## Dataset
A clean dataset was loaded from the `databricks_challenge` catalog.
Only numerical columns were used to keep the workflow simple and interpretable.

**Target variable:** Revenue  
**Features used:**  
- Quantity (Qty)  
- Customer Age (Age)  

---

## Feature Engineering
- Selected relevant numerical columns
- Cast columns to appropriate data types
- Removed null values
- Combined features using `VectorAssembler`

This ensured a clean feature vector suitable for Spark ML models.

---

## Models Trained
Three different regression models were trained independently:

1. **Linear Regression**  
2. **Decision Tree Regressor**  
3. **Random Forest Regressor**

Each model was trained using the same train-test split to ensure a fair comparison.

---

## Spark ML Pipeline
A separate Spark ML Pipeline was created for each model.
The pipeline approach makes the workflow modular, reusable, and production-ready.

---

## Model Evaluation
- The dataset was split into 80% training and 20% testing data
- Models were evaluated using **RMSE (Root Mean Squared Error)**
- Predictions were generated on the test dataset

---

## MLflow Experiment Tracking
- A new MLflow experiment was created specifically for Day 13
- Each model run was logged as a separate MLflow run
- Logged information included:
  - Model name
  - Feature set
  - RMSE metric

MLflow UI was used to visually compare all model runs.

---

## Model Comparison & Selection
All three models were compared side-by-side in the MLflow UI based on RMSE.

**Result:**  
The **Random Forest Regressor** achieved the lowest RMSE and was selected as the
best-performing model.

---

## Key Learnings
- Training multiple models provides better insight than relying on a single approach
- MLflow makes experiment comparison transparent and reproducible
- Spark ML Pipelines help structure ML workflows cleanly
- Model selection should always be based on evaluation metrics, not assumptions

---

## Tools & Technologies
- Databricks Community Edition  
- Apache Spark (PySpark ML)  
- MLflow  

---

## Outcome
By the end of Day 13, multiple models were successfully trained, tracked,
compared, and evaluated using MLflow, demonstrating a complete
model comparison workflow in Databricks.
