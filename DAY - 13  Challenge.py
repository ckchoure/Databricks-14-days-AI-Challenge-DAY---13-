# Databricks notebook source
# Load raw dataset 
data = spark.table("databricks_challenge.vrinda_store.vrinda_store_data")

data.show(5)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

# Select and cast required columns
feature_df_13 = data.select(
    col("Qty").cast("int").alias("qty"),
    col("Age").cast("int").alias("age"),
    col("Amount").cast("double").alias("label")
).dropna()

# Combine features into a single vector
assembler = VectorAssembler(
    inputCols=["qty", "age"],
    outputCol="features"
)

final_df_13 = assembler.transform(feature_df_13).select("features", "label")

final_df_13.show(5)

# COMMAND ----------

# Split data into training and testing sets
train_df, test_df = final_df_13.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor
)

# Model 1: Linear Regression
lr = LinearRegression()

# Model 2: Decision Tree Regressor
dt = DecisionTreeRegressor(maxDepth=5)

# Model 3: Random Forest Regressor
rf = RandomForestRegressor(numTrees=50, maxDepth=5)

# COMMAND ----------

from pyspark.ml import Pipeline

# Create pipelines for each model
pipelines = {
    "LinearRegression": Pipeline(stages=[lr]),
    "DecisionTree": Pipeline(stages=[dt]),
    "RandomForest": Pipeline(stages=[rf])
}

# COMMAND ----------

import mlflow

# Create a fresh MLflow experiment for Day 13
mlflow.set_experiment("/Shared/day_13_model_comparison")

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# RMSE evaluator
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# Train each model and log results
for model_name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=model_name):

        # Train model
        model = pipeline.fit(train_df)

        # Generate predictions
        predictions = model.transform(test_df)

        # Evaluate model performance
        rmse = evaluator.evaluate(predictions)

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("features", "qty, age")
        mlflow.log_metric("rmse", rmse)

        print(f"{model_name} RMSE: {rmse}")
