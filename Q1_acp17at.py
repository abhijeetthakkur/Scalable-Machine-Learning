import os
import subprocess
import numpy as np
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import Binarizer
from pyspark.ml.classification import LogisticRegression
import time



spark = SparkSession.builder \
    .master("local[20]") \
    .appName("ML_Assignment2") \
    .getOrCreate()

df = spark.read.option("header", "false").csv("../Data/HIGGS.csv.gz")
df = df.cache()


df1 = df.sample(False,0.25,1234)
df=df.repartition(10).cache()

changedTypedf = df1.withColumn("_c0", df1["_c0"].cast(DoubleType()))
changedTypedf1 = changedTypedf.withColumn("_c1", changedTypedf["_c1"].cast(DoubleType()))
changedTypedf2 = changedTypedf1.withColumn("_c2", changedTypedf1["_c2"].cast(DoubleType()))
changedTypedf3 = changedTypedf2.withColumn("_c3", changedTypedf2["_c3"].cast(DoubleType()))
changedTypedf4 = changedTypedf3.withColumn("_c4", changedTypedf3["_c4"].cast(DoubleType()))
changedTypedf5 = changedTypedf4.withColumn("_c5", changedTypedf4["_c5"].cast(DoubleType()))
changedTypedf6 = changedTypedf5.withColumn("_c6", changedTypedf5["_c6"].cast(DoubleType()))
changedTypedf7 = changedTypedf6.withColumn("_c7", changedTypedf6["_c7"].cast(DoubleType()))
changedTypedf8 = changedTypedf7.withColumn("_c8", changedTypedf7["_c8"].cast(DoubleType()))
changedTypedf9 = changedTypedf8.withColumn("_c9", changedTypedf8["_c9"].cast(DoubleType()))
changedTypedf10 = changedTypedf9.withColumn("_c10", changedTypedf9["_c10"].cast(DoubleType()))
changedTypedf11 = changedTypedf10.withColumn("_c11", changedTypedf10["_c11"].cast(DoubleType()))
changedTypedf12 = changedTypedf11.withColumn("_c12", changedTypedf11["_c12"].cast(DoubleType()))
changedTypedf13 = changedTypedf12.withColumn("_c13", changedTypedf12["_c13"].cast(DoubleType()))
changedTypedf14 = changedTypedf13.withColumn("_c14", changedTypedf13["_c14"].cast(DoubleType()))
changedTypedf15 = changedTypedf14.withColumn("_c15", changedTypedf14["_c15"].cast(DoubleType()))
changedTypedf16 = changedTypedf15.withColumn("_c16", changedTypedf15["_c16"].cast(DoubleType()))
changedTypedf17 = changedTypedf16.withColumn("_c17", changedTypedf16["_c17"].cast(DoubleType()))
changedTypedf18 = changedTypedf17.withColumn("_c18", changedTypedf17["_c18"].cast(DoubleType()))
changedTypedf19 = changedTypedf18.withColumn("_c19", changedTypedf18["_c19"].cast(DoubleType()))
changedTypedf20 = changedTypedf19.withColumn("_c20", changedTypedf19["_c20"].cast(DoubleType()))
changedTypedf21 = changedTypedf20.withColumn("_c21", changedTypedf20["_c21"].cast(DoubleType()))
changedTypedf22 = changedTypedf21.withColumn("_c22", changedTypedf21["_c22"].cast(DoubleType()))
changedTypedf23 = changedTypedf22.withColumn("_c23", changedTypedf22["_c23"].cast(DoubleType()))
changedTypedf24 = changedTypedf23.withColumn("_c24", changedTypedf23["_c24"].cast(DoubleType()))
changedTypedf25 = changedTypedf24.withColumn("_c25", changedTypedf24["_c25"].cast(DoubleType()))
changedTypedf26 = changedTypedf25.withColumn("_c26", changedTypedf25["_c26"].cast(DoubleType()))
changedTypedf27 = changedTypedf26.withColumn("_c27", changedTypedf26["_c27"].cast(DoubleType()))
changedTypedf28 = changedTypedf27.withColumn("_c28", changedTypedf27["_c28"].cast(DoubleType()))

assembler = VectorAssembler(
    inputCols=["_c1", "_c2", "_c3","_c4","_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28"],
    outputCol="feature")

output = assembler.transform(changedTypedf28)


f1 = output.drop('_c1')
f2 = f1.drop('_c2')
f3 = f2.drop('_c3')
f4 = f3.drop('_c4')
f5 = f4.drop('_c5')
f6 = f5.drop('_c6')
f7 = f6.drop('_c7')
f8 = f7.drop('_c8')
f9 = f8.drop('_c9')
f10 = f9.drop('_c10')
f11 = f10.drop('_c11')
f12 = f11.drop('_c12')
f13 = f12.drop('_c13')
f14 = f13.drop('_c14')
f15 = f14.drop('_c15')
f16 = f15.drop('_c16')
f17 = f16.drop('_c17')
f18 = f17.drop('_c18')
f19 = f18.drop('_c19')
f20 = f19.drop('_c20')
f21 = f20.drop('_c21')
f22 = f21.drop('_c22')
f23 = f22.drop('_c23')
f24 = f23.drop('_c24')
f25 = f24.drop('_c25')
f26 = f25.drop('_c26')
f27 = f26.drop('_c27')
f28 = f27.drop('_c28')

dff = f28.selectExpr("_c0 as label","feature as features")

(trainingData, testData) = dff.randomSplit([0.7, 0.3], 50)

print("DecisionTreeClassifier starting....................")

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[dt])
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [30, 31, 32]) \
    .addGrid(dt.impurity, ["gini", "entropy"])\
    .build()

evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
crossval_classifier = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2,parallelism = 3)
cvModel_classifier = crossval_classifier.fit(trainingData)
prediction_classifier = cvModel_classifier.transform(testData)

accuracy = evaluator.evaluate(prediction_classifier)
print("Accuracy of DecisionTreeClassifier = %g " % accuracy)


crossval_classifier_area_under_curve = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)
cvModel_classifier_area_under_curve = crossval_classifier_area_under_curve.fit(trainingData)
prediction_classifier_area_under_curve = cvModel_classifier_area_under_curve.transform(testData)
evaluator_classifier_area_under_curve  = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
classifier_area_under_curve = evaluator_classifier_area_under_curve.evaluate(prediction_classifier_area_under_curve)
print("Area Under the curve for Decision Tree Classifier",classifier_area_under_curve)
print("Printing Parameter for DecisionTreeClassifier")
bestPipeline = cvModel_classifier.bestModel
bestclassifierModel = bestPipeline.stages[0]
bestParamsclassifier = bestclassifierModel.extractParamMap()
for x in bestParamsclassifier:
    print (x.name,bestParamsclassifier[x])
print("/n")

print("Parameter for classification")
maxDepth = bestclassifierModel._java_obj.getMaxDepth()
print("Best max depth")
print(maxDepth)
print("Best impurity")
impurity = bestclassifierModel._java_obj.getImpurity()
print(impurity)
print("Best maxbins")
maxBins = bestclassifierModel._java_obj.getMaxBins()
print(maxBins)

print("DecisionTreeRegression starting.........................")

evaluator_regressor = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
dt1 = DecisionTreeRegressor(labelCol="label", featuresCol="features")
pipeline_regressor = Pipeline(stages=[dt1])
paramGrid_regressor = ParamGridBuilder() \
    .addGrid(dt1.maxDepth, [5, 10, 30]) \
    .addGrid(dt1.maxBins, [20, 35, 40]) \
    .build()
crossval_regressor = CrossValidator(estimator=pipeline_regressor,
                          estimatorParamMaps=paramGrid_regressor,
                          evaluator=evaluator_regressor,
                          numFolds=2,parallelism = 3)

cvModel_regressor = crossval_regressor.fit(trainingData)
prediction_regressor = cvModel_regressor.transform(testData)
binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="binarized_prediction")
binarizedDataFrame = binarizer.transform(prediction_regressor)
P1 = binarizedDataFrame.drop('prediction')
binarizedDataFrame_1 = P1.withColumnRenamed("binarized_prediction", "prediction")
evaluator_regressor1 = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_regressor = evaluator_regressor1.evaluate(binarizedDataFrame_1)
print("Accuracy of DecisionTreeRegressor= %g " % accuracy_regressor)

evaluator_regressor_area_under_curve =BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy_regressor_area_under_curve = evaluator_regressor_area_under_curve.evaluate(prediction_regressor)
print("Area Under the curve on Decision Trees for Regression ",accuracy_regressor_area_under_curve)
print("Printing Parameter for DecisionTreeRegressor")
bestPipelineregressor = cvModel_regressor.bestModel
bestRegressionModel = bestPipelineregressor.stages[0]
bestParamsRegression = bestRegressionModel.extractParamMap()
for x in bestParamsRegression:
    print(x.name, bestParamsRegression[x])

print("Parameter for regression")
maxDepth_reg = bestRegressionModel._java_obj.getMaxDepth()
print("Best max depth")
print(maxDepth_reg)
print("Best maxbins")
maxBins_reg = bestRegressionModel._java_obj.getMaxBins()
print(maxBins_reg)

print("Logistic Regression starting......................")

evaluator_logistic = MulticlassClassificationEvaluator \
    (labelCol="label", predictionCol="prediction", metricName="accuracy")

lr = LogisticRegression(featuresCol='features', labelCol='label')
pipeline_logistic = Pipeline(stages=[lr])
paramGrid_logistic = ParamGridBuilder() \
    .addGrid(lr.maxIter, [5, 10, 30]) \
    .addGrid(lr.regParam, [0.3, 0.6, 0.9]) \
    .addGrid(lr.elasticNetParam, [0.6, 0.8, 1.0]) \
    .build()
crossval_logistic = CrossValidator(estimator=pipeline_logistic,
                                   estimatorParamMaps=paramGrid_logistic,
                                   evaluator=evaluator_logistic,
                                   numFolds=2,parallelism = 3)
lrModel1 = crossval_logistic.fit(trainingData)
prediction_logistic = lrModel1.transform(testData)
accuracy_logistic = evaluator_logistic.evaluate(prediction_logistic)
print("Accuracy of Logistic Regression= %g " % accuracy_logistic)
print("Printing Parameter")
bestPipelineLR = lrModel1.bestModel
bestLRModel = bestPipelineLR.stages[0]
bestParamsLR = bestLRModel.extractParamMap()
for x in bestParamsLR:
    print (x.name,bestParamsLR[x])
print("Parameter for logistic")
maxIter = bestLRModel._java_obj.getMaxIter()
print("Best max Iter")
print(maxIter)
print("Best regParam")
regParam = bestLRModel._java_obj.getRegParam()
print(regParam)
print("Best elasticNetParam")
elasticNetParam = bestLRModel._java_obj.getElasticNetParam()
print(elasticNetParam)
evaluator_logistic_area_under_curve = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy_logistic_area_under_curve = evaluator_regressor_area_under_curve.evaluate(prediction_logistic)
print("Area Under the curve for d Logistic Regression",accuracy_logistic_area_under_curve)


#####################################100 percent data########################################################################

df = spark.read.option("header", "false").csv("../Data/HIGGS.csv.gz")
df1 = df.cache()
df=df.repartition(10).cache()


changedTypedf = df1.withColumn("_c0", df1["_c0"].cast(DoubleType()))
changedTypedf1 = changedTypedf.withColumn("_c1", changedTypedf["_c1"].cast(DoubleType()))
changedTypedf2 = changedTypedf1.withColumn("_c2", changedTypedf1["_c2"].cast(DoubleType()))
changedTypedf3 = changedTypedf2.withColumn("_c3", changedTypedf2["_c3"].cast(DoubleType()))
changedTypedf4 = changedTypedf3.withColumn("_c4", changedTypedf3["_c4"].cast(DoubleType()))
changedTypedf5 = changedTypedf4.withColumn("_c5", changedTypedf4["_c5"].cast(DoubleType()))
changedTypedf6 = changedTypedf5.withColumn("_c6", changedTypedf5["_c6"].cast(DoubleType()))
changedTypedf7 = changedTypedf6.withColumn("_c7", changedTypedf6["_c7"].cast(DoubleType()))
changedTypedf8 = changedTypedf7.withColumn("_c8", changedTypedf7["_c8"].cast(DoubleType()))
changedTypedf9 = changedTypedf8.withColumn("_c9", changedTypedf8["_c9"].cast(DoubleType()))
changedTypedf10 = changedTypedf9.withColumn("_c10", changedTypedf9["_c10"].cast(DoubleType()))
changedTypedf11 = changedTypedf10.withColumn("_c11", changedTypedf10["_c11"].cast(DoubleType()))
changedTypedf12 = changedTypedf11.withColumn("_c12", changedTypedf11["_c12"].cast(DoubleType()))
changedTypedf13 = changedTypedf12.withColumn("_c13", changedTypedf12["_c13"].cast(DoubleType()))
changedTypedf14 = changedTypedf13.withColumn("_c14", changedTypedf13["_c14"].cast(DoubleType()))
changedTypedf15 = changedTypedf14.withColumn("_c15", changedTypedf14["_c15"].cast(DoubleType()))
changedTypedf16 = changedTypedf15.withColumn("_c16", changedTypedf15["_c16"].cast(DoubleType()))
changedTypedf17 = changedTypedf16.withColumn("_c17", changedTypedf16["_c17"].cast(DoubleType()))
changedTypedf18 = changedTypedf17.withColumn("_c18", changedTypedf17["_c18"].cast(DoubleType()))
changedTypedf19 = changedTypedf18.withColumn("_c19", changedTypedf18["_c19"].cast(DoubleType()))
changedTypedf20 = changedTypedf19.withColumn("_c20", changedTypedf19["_c20"].cast(DoubleType()))
changedTypedf21 = changedTypedf20.withColumn("_c21", changedTypedf20["_c21"].cast(DoubleType()))
changedTypedf22 = changedTypedf21.withColumn("_c22", changedTypedf21["_c22"].cast(DoubleType()))
changedTypedf23 = changedTypedf22.withColumn("_c23", changedTypedf22["_c23"].cast(DoubleType()))
changedTypedf24 = changedTypedf23.withColumn("_c24", changedTypedf23["_c24"].cast(DoubleType()))
changedTypedf25 = changedTypedf24.withColumn("_c25", changedTypedf24["_c25"].cast(DoubleType()))
changedTypedf26 = changedTypedf25.withColumn("_c26", changedTypedf25["_c26"].cast(DoubleType()))
changedTypedf27 = changedTypedf26.withColumn("_c27", changedTypedf26["_c27"].cast(DoubleType()))
changedTypedf28 = changedTypedf27.withColumn("_c28", changedTypedf27["_c28"].cast(DoubleType()))


assembler = VectorAssembler(
    inputCols=["_c1", "_c2", "_c3","_c4","_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28"],
    outputCol="feature")

output = assembler.transform(changedTypedf28)


f1 = output.drop('_c1')
f2 = f1.drop('_c2')
f3 = f2.drop('_c3')
f4 = f3.drop('_c4')
f5 = f4.drop('_c5')
f6 = f5.drop('_c6')
f7 = f6.drop('_c7')
f8 = f7.drop('_c8')
f9 = f8.drop('_c9')
f10 = f9.drop('_c10')
f11 = f10.drop('_c11')
f12 = f11.drop('_c12')
f13 = f12.drop('_c13')
f14 = f13.drop('_c14')
f15 = f14.drop('_c15')
f16 = f15.drop('_c16')
f17 = f16.drop('_c17')
f18 = f17.drop('_c18')
f19 = f18.drop('_c19')
f20 = f19.drop('_c20')
f21 = f20.drop('_c21')
f22 = f21.drop('_c22')
f23 = f22.drop('_c23')
f24 = f23.drop('_c24')
f25 = f24.drop('_c25')
f26 = f25.drop('_c26')
f27 = f26.drop('_c27')
f28 = f27.drop('_c28')

dff = f28.selectExpr("_c0 as label","feature as features")
(trainingData, testData) = dff.randomSplit([0.7, 0.3], 50)

print("DecisionTreeClassifier starting............................")
time_classifier = time.time()
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",maxDepth =maxDepth,maxBins = maxBins,impurity = impurity)
model_dt = dt.fit(trainingData)
prediction_dt = model_dt.transform(testData)
time_classifier1 = time.time()-time_classifier
print("Time Taken by DecisionTreeClassifier",time_classifier1)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
print("Printing accuracy DecisionTreeClassifier")
accuracy = evaluator.evaluate(prediction_dt)
print("Accuracy DecisionTreeClassifier = %g " % accuracy)
evaluator_classifier_area_under_curve  = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
classifier_area_under_curve = evaluator_classifier_area_under_curve.evaluate(prediction_dt)
print("Area Under the curve for Decision Tree Classifier",classifier_area_under_curve)

print("Top 3 features for DecisionTreeClassifier")
fi_classification = model_dt.featureImportances
#print(fi_classification)
ncolumns = len(df1.columns)
imp_feat = np.zeros(ncolumns-1)
imp_feat[fi_classification.indices] = fi_classification.values
#print(fi.values)
Index_classification = (-imp_feat).argsort()[:3]
schemaNames = df1.schema.names
for i in Index_classification:
    print("Top features",schemaNames[i+1])

print("DecisionTreeRegressor starting.........................")
time_regressor = time.time()
evaluator_regressor = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
dt1 = DecisionTreeRegressor(labelCol="label", featuresCol="features",maxBins=maxBins_reg,maxDepth=maxDepth_reg)
model_dt1 = dt1.fit(trainingData)
prediction_dt1=model_dt1.transform(testData)
time_regressor1 = time.time()-time_regressor
print("Time Taken by DecisionTreeRegressor",time_regressor1)
binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="binarized_prediction")
binarizedDataFrame = binarizer.transform(prediction_dt1)
P1 = binarizedDataFrame.drop('prediction')
binarizedDataFrame_1 = P1.withColumnRenamed("binarized_prediction", "prediction")
evaluator_regressor1 = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_regressor = evaluator_regressor1.evaluate(binarizedDataFrame_1)
print("Accuracy DecisionTreeRegressor = %g " % accuracy_regressor)
evaluator_regressor_area_under_curve = BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy_regressor_area_under_curve = evaluator_regressor_area_under_curve.evaluate(prediction_dt1)
print("Area Under the curve on Decision Trees for Regression ",accuracy_regressor_area_under_curve)

print("Top 3 features for DecisionTreeRegressor")
fi_regression = model_dt.featureImportances
#print(fi_regression)
ncolumns = len(df1.columns)
imp_feat_regression = np.zeros(ncolumns-1)
imp_feat_regression[fi_regression.indices] = fi_regression.values
#print(fi_regression.values)
Index_regression = (-imp_feat_regression).argsort()[:3]
schemaNames = df1.schema.names
for i in Index_regression:
    print("Top features",schemaNames[i+1])

print("Logistic Regression starting.......................")
time_logistic = time.time()
evaluator_logistic = MulticlassClassificationEvaluator \
    (labelCol="label", predictionCol="prediction", metricName="accuracy")
lr = LogisticRegression(featuresCol='features', labelCol='label',maxIter = maxIter,regParam = regParam)
model_lr = lr.fit(trainingData)
prediction_lr=model_lr.transform(testData)
time_logistic1 = time.time()-time_logistic
print("Time Taken by Logistic Regression",time_logistic1)
accuracy_logistic = evaluator_logistic.evaluate(prediction_lr)
print("Accuracy LogisticRegression = %g " % accuracy_logistic)
evaluator_logistic_area_under_curve = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy_logistic_area_under_curve = evaluator_regressor_area_under_curve.evaluate(prediction_lr)
print("Area Under the curve for d Logistic Regression",accuracy_logistic_area_under_curve)

print("Top 3 features for LogisticRegression")
imp_logistic = model_lr.coefficients.values
Index_sort = np.argsort(imp_logistic)
top_3_features = Index_sort[:3]
schemaNames = df1.schema.names
for i in top_3_features:
    print(schemaNames[i+1])

print("End of code............................")
spark.stop()