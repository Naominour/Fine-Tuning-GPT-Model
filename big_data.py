from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("ReadLargeCSV") \
    .getOrCreate()

# Read the CSV file
df = spark.read.csv('C:/Users/H.Naeeme/Downloads/full_data.csv/full_data.csv', header=True, inferSchema=True)

# Show the first few rows
print(df.head())