
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

print("Print0")
spark = SparkSession.builder \
    .master("local[20]") \
    .appName("ML_ques") \
    .getOrCreate()

sc = spark.context
sc.setLogLevel("WARN")


dff = spark.read.csv("../Data/train_set.csv", header=True, mode="DROPMALFORMED")
dfff = dff.cache()
dfff=dfff.repartition(10).cache()

for x in dfff.columns:
    dfff = dfff.filter((dfff[x] != '?'))

changedTypedf1 = dfff.withColumn("Row_ID", dfff["Row_ID"].cast(DoubleType()))
changedTypedf2 = changedTypedf1.withColumn("Household_ID", changedTypedf1["Household_ID"].cast(DoubleType()))
changedTypedf3 = changedTypedf2.withColumn("Vehicle", changedTypedf2["Vehicle"].cast(DoubleType()))
changedTypedf4 = changedTypedf3.withColumn("Calendar_Year", changedTypedf3["Calendar_Year"].cast(DoubleType()))
changedTypedf5 = changedTypedf4.withColumn("Model_Year", changedTypedf4["Model_Year"].cast(DoubleType()))
changedTypedf6 = StringIndexer(inputCol="Blind_Make", outputCol="Blind_Make1").fit(changedTypedf5)
changedTypedf7 = changedTypedf6.transform(changedTypedf5)
changedTypedf7=changedTypedf7.drop("Blind_Make")
changedTypedf8 = StringIndexer(inputCol="Blind_Model", outputCol="Blind_Model1").fit(changedTypedf7)
changedTypedf9 = changedTypedf8.transform(changedTypedf7)
changedTypedf9=changedTypedf9.drop("Blind_Model")
changedTypedf10 = StringIndexer(inputCol="Blind_Submodel", outputCol="Blind_Submodel1").fit(changedTypedf9)
changedTypedf11 = changedTypedf10.transform(changedTypedf9)
changedTypedf11=changedTypedf11.drop("Blind_Submodel")
changedTypedf12 = StringIndexer(inputCol="Cat1", outputCol="Cat1_1").fit(changedTypedf11)
changedTypedf13 = changedTypedf12.transform(changedTypedf11)
changedTypedf13=changedTypedf13.drop("Cat1")
changedTypedf14 = StringIndexer(inputCol="Cat2", outputCol="Cat2_1").fit(changedTypedf13)
changedTypedf15 = changedTypedf14.transform(changedTypedf13)
changedTypedf15=changedTypedf15.drop("Cat2")
changedTypedf16 = StringIndexer(inputCol="Cat3", outputCol="Cat3_1").fit(changedTypedf15)
changedTypedf17 = changedTypedf16.transform(changedTypedf15)
changedTypedf17=changedTypedf17.drop("Cat3")
changedTypedf18 = StringIndexer(inputCol="Cat4", outputCol="Cat4_1").fit(changedTypedf17)
changedTypedf19 = changedTypedf18.transform(changedTypedf17)
changedTypedf19=changedTypedf19.drop("Cat4")
changedTypedf20 = StringIndexer(inputCol="Cat5", outputCol="Cat5_1").fit(changedTypedf19)
changedTypedf21 = changedTypedf20.transform(changedTypedf19)
changedTypedf21=changedTypedf21.drop("Cat5")
changedTypedf22 = StringIndexer(inputCol="Cat6", outputCol="Cat6_1").fit(changedTypedf21)
changedTypedf23 = changedTypedf22.transform(changedTypedf21)
changedTypedf23=changedTypedf23.drop("Cat6")
changedTypedf24 = StringIndexer(inputCol="Cat7", outputCol="Cat7_1").fit(changedTypedf23)
changedTypedf25 = changedTypedf24.transform(changedTypedf23)
changedTypedf25=changedTypedf25.drop("Cat7")
changedTypedf26 = StringIndexer(inputCol="Cat8", outputCol="Cat8_1").fit(changedTypedf25)
changedTypedf27 = changedTypedf26.transform(changedTypedf25)
changedTypedf27=changedTypedf27.drop("Cat8")
changedTypedf28 = StringIndexer(inputCol="Cat9", outputCol="Cat9_1").fit(changedTypedf27)
changedTypedf29 = changedTypedf28.transform(changedTypedf27)
changedTypedf29=changedTypedf29.drop("Cat9")
changedTypedf30 = StringIndexer(inputCol="Cat10", outputCol="Cat10_1").fit(changedTypedf29)
changedTypedf31 = changedTypedf30.transform(changedTypedf29)
changedTypedf31=changedTypedf31.drop("Cat10")
changedTypedf32 = StringIndexer(inputCol="Cat11", outputCol="Cat11_1").fit(changedTypedf31)
changedTypedf33 = changedTypedf32.transform(changedTypedf31)
changedTypedf33=changedTypedf33.drop("Cat11")
changedTypedf34 = StringIndexer(inputCol="Cat12", outputCol="Cat12_1").fit(changedTypedf33)
changedTypedf35 = changedTypedf34.transform(changedTypedf33)
changedTypedf35=changedTypedf35.drop("Cat12")
changedTypedf36 = changedTypedf35.withColumn("OrdCat", changedTypedf35["OrdCat"].cast(DoubleType()))
changedTypedf37 = changedTypedf36.withColumn("Var1", changedTypedf36["Var1"].cast(DoubleType()))
changedTypedf38 = changedTypedf37.withColumn("Var2", changedTypedf37["Var2"].cast(DoubleType()))
changedTypedf39 = changedTypedf38.withColumn("Var3", changedTypedf38["Var3"].cast(DoubleType()))
changedTypedf40 = changedTypedf39.withColumn("Var4", changedTypedf39["Var4"].cast(DoubleType()))
changedTypedf41 = changedTypedf40.withColumn("Var5", changedTypedf40["Var5"].cast(DoubleType()))
changedTypedf42 = changedTypedf41.withColumn("Var6", changedTypedf41["Var6"].cast(DoubleType()))
changedTypedf43 = changedTypedf42.withColumn("Var7", changedTypedf42["Var7"].cast(DoubleType()))
changedTypedf44 = changedTypedf43.withColumn("Var8", changedTypedf43["Var8"].cast(DoubleType()))
changedTypedf45 = StringIndexer(inputCol="NVCat", outputCol="NVCat_1").fit(changedTypedf44)
changedTypedf46 = changedTypedf45.transform(changedTypedf44)
changedTypedf46=changedTypedf46.drop("NVCat")
changedTypedf47 = changedTypedf46.withColumn("NVVar1", changedTypedf46["NVVar1"].cast(DoubleType()))
changedTypedf48 = changedTypedf47.withColumn("NVVar2", changedTypedf47["NVVar2"].cast(DoubleType()))
changedTypedf49 = changedTypedf48.withColumn("NVVar3", changedTypedf48["NVVar3"].cast(DoubleType()))
changedTypedf50 = changedTypedf49.withColumn("NVVar4", changedTypedf49["NVVar4"].cast(DoubleType()))
changedTypedf51 = changedTypedf50.withColumn("Claim_Amount", changedTypedf50["Claim_Amount"].cast(DoubleType()))
changedTypedf52 = changedTypedf51.drop("Row_ID")
changedTypedf53 = changedTypedf52.drop("Household_ID")
changedTypedf54 = changedTypedf53.drop("Vehicle")
assembler = VectorAssembler(
    inputCols=["Calendar_Year","Model_Year","OrdCat","Var1","Var2","Var3","Var4","Var5","Var6","Var7","Var8","NVVar1","NVVar2","NVVar3","NVVar4","Blind_Make1","Blind_Model1","Blind_Submodel1","Cat1_1","Cat2_1","Cat3_1","Cat3_1","Cat4_1","Cat5_1","Cat6_1","Cat7_1","Cat8_1","Cat9_1","Cat10_1","Cat11_1","Cat12_1","NVCat_1"],
    outputCol="feature")

output = assembler.transform(changedTypedf54)
output = output.cache()

f1 = output.drop('Calendar_Year')
f2 = f1.drop('Model_Year')
f3 = f2.drop('OrdCat')
f4 = f3.drop('Var1')
f5 = f4.drop('Var2')
f6 = f5.drop('Var3')
f7 = f6.drop('Var4')
f8 = f7.drop('Var5')
f9 = f8.drop('Var6')
f10 = f9.drop('Var7')
f11 = f10.drop('Var8')
f12 = f11.drop('NVVar1')
f13 = f12.drop('NVVar2')
f14 = f13.drop('NVVar3')
f15 = f14.drop('NVVar4')
f16 = f15.drop('Blind_Make1')
f17 = f16.drop('Blind_Model1')
f18 = f17.drop('Blind_Submodel1')
f19 = f18.drop('Cat1_1')
f20 = f19.drop('Cat2_1')
f21 = f20.drop('Cat3_1')
f22 = f21.drop('Cat4_1')
f23 = f22.drop('Cat5_1')
f24 = f23.drop('Cat6_1')
f25 = f24.drop('Cat7_1')
f26 = f25.drop('Cat8_1')
f27 = f26.drop('Cat9_1')
f28 = f27.drop('Cat10_1')
f29 = f28.drop('Cat11_1')
f30 = f29.drop('Cat12_1')
f31 = f30.drop('NVCat_1')
df2 = f31.selectExpr("Claim_Amount as label","feature as features")


assembler1 = VectorAssembler(
    inputCols=["label"],
    outputCol="label1")

output1 = assembler1.transform(df2)
output1 = output1.cache()
f32 = output1.drop("label")
scaler = MinMaxScaler(inputCol="label1", outputCol="label")
scalerModel = scaler.fit(f32)
scaledData = scalerModel.transform(f32)
element=udf(lambda v:float(v[0]),FloatType())
new = scaledData.withColumn('label', element('label'))
(trainingData, testData) = new.randomSplit([0.7, 0.3], 50)
lr = LinearRegression(featuresCol = "features", labelCol = "label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainingData)
trainingSummary = lrModel.summary
print("Question 2.2(a).............")
print("RMSE for training data: %f" % trainingSummary.rootMeanSquaredError)
predict = lrModel.transform(testData)
evaluator_rmse = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predict)
print("RMSE for test data = %g " % rmse)


spark.stop()
