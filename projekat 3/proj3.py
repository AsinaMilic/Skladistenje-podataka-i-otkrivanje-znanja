from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Rice Classificator") \
    .getOrCreate()

path = "./Rice.csv"

# Učitavanje podataka
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)

# Pretvaranje string etiketa u numerički format
indexer = StringIndexer(inputCol="Class", outputCol="label")
indexedData = indexer.fit(data).transform(data)

# Odabir relevantnih atributa i pretvaranje u vektor
assembler = VectorAssembler(inputCols=["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea",
                                       "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation"],
                            outputCol="features")
assembledData = assembler.transform(indexedData)

# Deljenje podataka na skup za treniranje i skup za testiranje
(trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=13156123)

# Treniranje modela logističke regresije
lr = LogisticRegression()
lrModel = lr.fit(trainingData)

# Treniranje modela Naivnog Bayesa
nb = NaiveBayes()
nbModel = nb.fit(trainingData)

# Treniranje SVM modela
svm = LinearSVC(maxIter=10)
svmModel = svm.fit(trainingData)

# Treniranje modela drveća odlučivanja
dt = DecisionTreeClassifier()
dtModel = dt.fit(trainingData)

# Predviđanje na testnom skupu
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

# Ispis rezultata
print("Logistic Regression Accuracy: " + str(lrAccuracy))
print("Naive Bayes Accuracy: " + str(nbAccuracy))
print("SVM Accuracy: " + str(svmAccuracy))
print("Decision Tree Accuracy: " + str(dtAccuracy))

# Zatvaranje Spark sesije
spark.stop()
