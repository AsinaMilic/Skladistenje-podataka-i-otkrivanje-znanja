from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum,stddev,stddev_samp,expr,when, udf
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

spark = SparkSession.builder \
    .appName("Rice Classificator") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN") # ne zelim da mi se INFO logovi prikazuju u konzoli

path = "./Rice.csv"

###########################  1) Učitavanje podataka  #######################################################
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)
#############################################################################################################


########################  2) Preprocesranje (koristeci spark) ##############################################
data = data.drop("id")
print("\nID kolona uklonjena::")
data.show()

# Izbroj i izbaci iste redove (imaju sve vrednosti atributa iste)
duplicate_count = data.dropDuplicates().count() - data.count()
print(f"\nNumber of duplicate rows: {duplicate_count}\n")

print("\nKoliko polja u svakoj koloni su null?")
data.select([sum(col(column).isNull().cast("int")).alias(column) for column in data.columns]).show()

print("\nDa li vrednosti atributa class balansirane?")
data.groupBy("class").count().show()

# Eliminisemo "outliers" vrednosti za ove kolone
columns = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea","EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation"]

statistics = data.select([stddev_samp(column).alias(column) for column in columns]).first() # Calculate the mean and standard deviation for each column

# Define the lower and upper bounds for outliers (moze i 3)
lower_bounds = [(statistics[column] - 2 * statistics[column]) for column in columns]
upper_bounds = [(statistics[column] + 2 * statistics[column]) for column in columns]

# Filter out rows with values outside the bounds
print("\nFiltrirani podaci:")
data.filter(expr(" AND ".join([f"({column} >= {lower_bound} AND {column} <= {upper_bound})" for column, lower_bound, upper_bound in zip(columns, lower_bounds, upper_bounds)]))).show()

print("\nNormalizovani podaci:")
# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),15), DoubleType())
for i in columns:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_S")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    data = pipeline.fit(data).transform(data).withColumn(i+"_S", unlist(i+"_S")).drop(i+"_Vect")

# Drop the first 10 columns
drop_columns = data.columns[:10]
data = data.drop(*drop_columns)
data = data.select(*([col(c) for c in data.columns[1:]] + [col(data.columns[0])])) # Move the 11th column to the end
data.show()

#eventualno da koristim selectovane attribute
#############################################################################################################


# Pretvaranje string etiketa u numerički format
indexer = StringIndexer(inputCol="Class", outputCol="label")
indexedData = indexer.fit(data).transform(data)

# Odabir relevantnih atributa i pretvaranje u vektor
assembler = VectorAssembler(inputCols=["Area_S", "MajorAxisLength_S", "MinorAxisLength_S", "Eccentricity_S", "ConvexArea_S","EquivDiameter_S", "Extent_S", "Perimeter_S", "Roundness_S", "AspectRation_S"],
                           outputCol="features")
assembledData = assembler.transform(indexedData)

# Deljenje podataka na skup za treniranje i skup za testiranje
(trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=13156123)

##################################  3) Kreirati model/e mašinskog učenja  ####################################
lr = LogisticRegression()
nb = NaiveBayes()
svm = LinearSVC(maxIter=10)
dt = DecisionTreeClassifier()

# Treniranje
lrModel = lr.fit(trainingData) 
nbModel = nb.fit(trainingData) 
svmModel = svm.fit(trainingData) 
dtModel = dt.fit(trainingData)
##############################################################################################################

##################################   4)  Kreirani model/e testirati  #########################################
lrPredictions = lrModel.transform(testData)
nbPredictions = nbModel.transform(testData)
svmPredictions = svmModel.transform(testData)
dtPredictions = dtModel.transform(testData)

# Evaluacija modela
lrEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
nbEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
svmEvaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
dtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
lrAccuracy = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "accuracy"})
nbAccuracy = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "accuracy"})
svmAccuracy = svmEvaluator.evaluate(svmPredictions)
dtAccuracy = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "accuracy"})
##############################################################################################################


##################################  5) Sačuvati i prikazati rezultate.  ######################################
print("Logistic Regression Accuracy: " + str(lrAccuracy))
print("Naive Bayes Accuracy: " + str(nbAccuracy))
print("SVM Accuracy: " + str(svmAccuracy))
print("Decision Tree Accuracy: " + str(dtAccuracy))

spark.stop()
