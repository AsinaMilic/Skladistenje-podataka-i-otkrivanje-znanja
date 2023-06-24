from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, LogisticRegression, NaiveBayes, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum,stddev,stddev_samp,expr,when, udf
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics
import sys


PREPROCESS_DATA = True
columns = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea","EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation"]
columns_scaled = ["Area_S", "MajorAxisLength_S", "MinorAxisLength_S", "Eccentricity_S", "ConvexArea_S","EquivDiameter_S", "Extent_S", "Perimeter_S", "Roundness_S", "AspectRation_S"]


def classificator_evaluation(PredictionAndLabels, output_file):
    Metrics = MulticlassMetrics(PredictionAndLabels)
    ConfusionMatrix = Metrics.confusionMatrix()
    cm = ConfusionMatrix.toArray()
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    P = TP + FP
    N = FN + TN
    Ukupno = P + N

    accuracy = (TP + TN) / cm.sum() #Ukupno
    error_rate = 1 - accuracy
    sensitivity = TP/P #(TP) / (TP + FN)
    specificity = TN/N #(TN) / (TN + FP)
    precision = (TP) / (TP + FP)
    recall = TP/P
    F = (2*precision*recall) / (precision+recall)
    pe = (TP+FP)*(TP+FN)/N*N + (TN+FN)*(TN+FP)/N*N  #kappa 

    with open(output_file, 'w') as file:
        sys.stdout = file
        print("                 Rezultat klasifikacije           ")
        print("                      Da       Ne           Ukupno")
        print(f"Poznata    Da        {TP}    {FN}           {P}")
        print(f" klasa     Ne         {FP}    {TN}          {N}")
        print(f"         Ukupno      {P}    {N}          {Ukupno}")
        print("\n\n\n")
        print("Number of instances: ", Ukupno)
        print("Accuracy: ", accuracy)
        print("Error rate: ", error_rate)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        print("Precision: ", precision)
        print("recall", recall)
        print("F-score:", F)
        print("kappa: ", pe)
        print("Confusion Matrix:")
        for row in cm:
            print(" ".join(str(int(x)) for x in row))

    sys.stdout = sys.__stdout__


def preprocess_data(data):
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
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect") # VectorAssembler Transformation - Converting column to vector type
        scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_S") # MinMaxScaler Transformation
        pipeline = Pipeline(stages=[assembler, scaler])  # Pipeline of VectorAssembler and MinMaxScaler
        data = pipeline.fit(data).transform(data).withColumn(i+"_S", unlist(i+"_S")).drop(i+"_Vect")  # Fitting pipeline on dataframe

    # Drop the first 10 columns
    drop_columns = data.columns[:10]
    data = data.drop(*drop_columns)
    data = data.select(*([col(c) for c in data.columns[1:]] + [col(data.columns[0])])) # Move the 11th column to the end
    data.show()

    return data   


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Rice Classificator") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN") # ne zelim da mi se INFO logovi prikazuju u konzoli
    path = "./Rice.csv"

    ###########################  1) Učitavanje podataka  #######################################################
    raw_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)


    ########################  2) Preprocesranje (koristeci spark) ##############################################
    if (PREPROCESS_DATA):
        data = preprocess_data(raw_data)
     #eventualno da koristim selectovane attribute


    ##################################  3) Kreirati model/e mašinskog učenja  ####################################
    # Pretvaranje iz Class u label jer tako algoritmi traze
    indexer = StringIndexer(inputCol="Class", outputCol="label")
    indexedData = indexer.fit(data).transform(data)

    # Odabir relevantnih atributa i pretvaranje u vektor
    assembler = VectorAssembler(inputCols=columns_scaled if PREPROCESS_DATA else columns, outputCol="features")
    assembledData = assembler.transform(indexedData)

    # Deljenje podataka na skup za treniranje i skup za testiranje
    (trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=13156123) 

    lr = LogisticRegression()
    nb = NaiveBayes()
    svm = LinearSVC(maxIter=10)
    dt = DecisionTreeClassifier()
    gbt = GBTClassifier()

    # Treniranje
    lrModel = lr.fit(trainingData) 
    nbModel = nb.fit(trainingData) 
    svmModel = svm.fit(trainingData) 
    dtModel = dt.fit(trainingData)
    gbtModel = gbt.fit(trainingData)


    ##################################   4)  Kreirani model/e testirati  #########################################
    lrPredictions = lrModel.transform(testData)
    nbPredictions = nbModel.transform(testData)
    svmPredictions = svmModel.transform(testData)
    dtPredictions = dtModel.transform(testData)
    gbtPredictions = gbtModel.transform(testData)

    # Evaluacija modela
    lrEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    nbEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    svmEvaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    dtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    gbtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    lrAccuracy = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "accuracy"})
    nbAccuracy = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "accuracy"})
    svmAccuracy = svmEvaluator.evaluate(svmPredictions)
    dtAccuracy = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "accuracy"})
    gbtAccuracy = gbtEvaluator.evaluate(gbtPredictions, {gbtEvaluator.metricName: "accuracy"})

    lrPredictionAndLabels = lrPredictions.select("prediction", "label").rdd
    classificator_evaluation(lrPredictionAndLabels,"Logistic Regression.txt")
    nbPredictionAndLabels = nbPredictions.select("prediction", "label").rdd
    classificator_evaluation(nbPredictionAndLabels,"Naive Bayes.txt")
    svmPredictionAndLabels = svmPredictions.select("prediction", "label").rdd
    classificator_evaluation(svmPredictionAndLabels,"SVM.txt")
    dtPredictionAndLabels = dtPredictions.select("prediction", "label").rdd
    classificator_evaluation(dtPredictionAndLabels,"Decision Tree.txt")
    gbtPredictionAndLabels = gbtPredictions.select("prediction", "label").rdd
    classificator_evaluation(gbtPredictionAndLabels,"GBT.txt")                                          

    ##################################  5) Sačuvati i prikazati rezultate.  ######################################
    print("Logistic Regression Accuracy: " + str(lrAccuracy))
    print("Naive Bayes Accuracy: " + str(nbAccuracy))
    print("SVM Accuracy: " + str(svmAccuracy))
    print("Decision Tree Accuracy: " + str(dtAccuracy))
    print("Gradient Boosted Trees: " + str(gbtAccuracy))
    spark.stop()