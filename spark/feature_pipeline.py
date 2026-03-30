import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import pandas as pd

print("Starting Spark session...")

# Initialize Spark
spark = SparkSession.builder \
    .appName("DataDecayDetection") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version: {spark.version}")

# Load processed data
print("Loading processed dataset...")
df = spark.read.parquet("data/processed/structured_dataset.parquet")
print(f"Loaded {df.count():,} rows with {len(df.columns)} columns")

# Feature Engineering at Scale
print("Engineering features with Spark...")

# Window specs for time-series calculations per ticker
ticker_window_30 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-30, 0)
ticker_window_60 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-60, 0)

ticker_window_5  = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-5, 0)

# Price momentum features
df = df.withColumn("Momentum_5d",
    F.col("Close") / F.lag("Close", 5).over(
        Window.partitionBy("Ticker").orderBy("Date")

    ) - 1
)

df = df.withColumn("Momentum_20d",
    F.col("Close") / F.lag("Close", 20).over(
        Window.partitionBy("Ticker").orderBy("Date")
    ) - 1
)

# Rolling max drawdown (key risk feature)
df = df.withColumn("Rolling_Max_30d",
    F.max("Close").over(ticker_window_30)
)
df = df.withColumn("Drawdown",
    (F.col("Close") - F.col("Rolling_Max_30d")) / F.col("Rolling_Max_30d")
)


# Average volume ratio (current vol vs 30d average)
df = df.withColumn("Avg_Volume_30d",
    F.avg("Volume").over(ticker_window_30)
)
df = df.withColumn("Volume_Ratio",
    F.col("Volume") / (F.col("Avg_Volume_30d") + 1)
)

# Sector-level aggregations (cross-stock drift signal)
sector_window = Window.partitionBy("Sector", "Date")
df = df.withColumn("Sector_Avg_Return",
    F.avg("Daily_Return").over(sector_window)
)
df = df.withColumn("Sector_Avg_Volatility",
    F.avg("Volatility_30d").over(sector_window)
)

# Return vs sector mean (stock-specific drift signal)
df = df.withColumn("Return_vs_Sector",
    F.col("Daily_Return") - F.col("Sector_Avg_Return")
)


# Drop nulls from window calculations
df = df.dropna()


print(f"After feature engineering: {df.count():,} rows, {len(df.columns)} columns")

# Select final feature set
feature_cols = [
    "Daily_Return",
    "Volatility_30d",
    "Return_Zscore",
    "Volatility_Ratio",
    "Volume_Zscore",
    "Momentum_5d",
    "Momentum_20d",
    "Drawdown",
    "Volume_Ratio",
    "Sector_Avg_Return",
    "Sector_Avg_Volatility",
    "Return_vs_Sector",
    "Golden_Cross",
]




# Assemble & Scale features
print("Assembling and scaling features...")

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withStd=True,
    withMean=True
)

pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(df)
df_final = pipeline_model.transform(df)

print("Pipeline complete!")

# Save outputs
print("Saving outputs...")
os.makedirs("data/spark_output", exist_ok=True)

# Save full featured dataset
df_final.select(
    "Date", "Ticker", "Sector", "Window", "Drift_Label",
    "Close", "Daily_Return", "Volatility_30d",
    "Return_Zscore", "Volatility_Ratio", "Volume_Zscore",
    "Momentum_5d", "Momentum_20d", "Drawdown",
    "Volume_Ratio", "Sector_Avg_Return", "Sector_Avg_Volatility",
    "Return_vs_Sector", "Golden_Cross"
).toPandas().to_parquet("data/spark_output/featured_dataset.parquet", index=False)

print("\n✅ Spark pipeline complete!")
print(f"Final dataset saved to data/spark_output/featured_dataset.parquet")

spark.stop()
