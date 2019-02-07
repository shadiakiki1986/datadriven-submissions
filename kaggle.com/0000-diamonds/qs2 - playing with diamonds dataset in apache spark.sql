-- Databricks notebook source
DROP TABLE IF EXISTS diamonds;

-- https://docs.databricks.com/user-guide/tables.html#create-a-local-table
-- https://ggplot2.tidyverse.org/reference/diamonds.html
CREATE TABLE diamonds
USING csv
OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true", inferSchema "true")


-- COMMAND ----------

-- select min(price), mean(price), max(price) from diamonds -- limit 10
select * from diamonds limit 5

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import avg
-- MAGIC diamonds = spark.sql("SELECT color, carat, depth, price FROM diamonds")
-- MAGIC display(diamonds.select("color","price").groupBy("color").agg(avg("price")).sort("color"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print("corr(carat, price) = ", diamonds.corr("carat", "price"))
-- MAGIC print("corr(depth, price) = ", diamonds.corr("depth", "price"))
-- MAGIC print("corr(depth, carat) = ", diamonds.corr("depth", "carat"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC diamonds[["carat", "price"]]

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # https://www.instaclustr.com/support/documentation/apache-spark/spark-mllib-linear-regression-example/
-- MAGIC from pyspark.ml.feature import VectorAssembler
-- MAGIC 
-- MAGIC assembler1 = VectorAssembler(inputCols=["carat"], outputCol="features").transform(diamonds)
-- MAGIC assembler1.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression
-- MAGIC from pyspark.ml.regression import LinearRegression
-- MAGIC lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal", #weightCol="weight",
-- MAGIC                      featuresCol="features", labelCol="price")
-- MAGIC model = lr.fit(assembler1)
-- MAGIC print(model.coefficients, model.intercept)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC model.transform(assembler1).show() #.head(n=3) #.prediction

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #display(assembler1)
-- MAGIC assembler1.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 0.23 * 7756 - 2256

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Summarize the model over the training set and print out some metrics
-- MAGIC trainingSummary = model.summary
-- MAGIC print("numIterations: %d" % trainingSummary.totalIterations)
-- MAGIC print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
-- MAGIC #trainingSummary.residuals.show()
-- MAGIC print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
-- MAGIC print("r2: %f" % trainingSummary.r2)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC fig, ax = plt.subplots()
-- MAGIC trainingSummary.residuals.sort("residuals").toPandas()["residuals"].plot()
-- MAGIC display(fig)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC trainingSummary.residuals.sort('residuals', ascending=False).show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from  pyspark.sql.functions import abs
-- MAGIC df = trainingSummary.residuals
-- MAGIC df = df.withColumn('absres', abs(df.residuals))
-- MAGIC df = df.sort('absres', ascending=True)
-- MAGIC df.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df.filter(df.absres < 10).count(), df.filter(df.absres >= 10).count()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # https://stackoverflow.com/questions/46225587/how-to-bin-in-pyspark
-- MAGIC # https://stackoverflow.com/questions/42451189/how-to-calculate-the-counts-of-each-distinct-value-in-a-pyspark-dataframe#42452414
-- MAGIC from pyspark.ml.feature import Bucketizer
-- MAGIC bucketizer = Bucketizer(splits=[ 0, 100, 1000, float('Inf') ], inputCol="absres", outputCol="buckets")
-- MAGIC df_buck = bucketizer.transform(df)
-- MAGIC 
-- MAGIC df2 = df_buck.groupby('buckets').count()
-- MAGIC df3 = df2.toPandas()
-- MAGIC df2.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # https://docs.databricks.com/user-guide/visualizations/matplotlib-and-ggplot.html
-- MAGIC from matplotlib import pyplot as plt
-- MAGIC fig, ax = plt.subplots()
-- MAGIC df3.set_index("buckets")["count"].plot()
-- MAGIC display(fig)

-- COMMAND ----------

-- MAGIC %sh
-- MAGIC /databricks/python/bin/pip install plotnine matplotlib==2.2.2

-- COMMAND ----------

-- MAGIC %python
-- MAGIC diamonds = spark.sql("SELECT * FROM diamonds")
-- MAGIC diamonds.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # try again to build a model that fits the prices better using other features than just carat
-- MAGIC 
-- MAGIC # convert strings to one-hot encoding
-- MAGIC # https://stackoverflow.com/a/32278617/4126114
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator
-- MAGIC 
-- MAGIC indexer1 = StringIndexer(inputCol="cut", outputCol="cutIndex")
-- MAGIC indexer2 = StringIndexer(inputCol="color", outputCol="colorIndex")
-- MAGIC indexer3 = StringIndexer(inputCol="clarity", outputCol="clarityIndex")
-- MAGIC inputs = [indexer1.getOutputCol(), indexer2.getOutputCol(), indexer3.getOutputCol()]
-- MAGIC encoder = OneHotEncoderEstimator(inputCols=inputs, outputCols=["cutVec", "colorVec", "clarityVec"])
-- MAGIC pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, encoder])
-- MAGIC diamonds2 = pipeline.fit(diamonds).transform(diamonds)
-- MAGIC 
-- MAGIC diamonds2.show()

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %python
-- MAGIC # https://www.instaclustr.com/support/documentation/apache-spark/spark-mllib-linear-regression-example/
-- MAGIC from pyspark.ml.feature import VectorAssembler
-- MAGIC 
-- MAGIC assembler2 = VectorAssembler(
-- MAGIC   inputCols=["carat", "depth", "table", "cutVec", "colorVec", "clarityVec", "x", "y", "z"], 
-- MAGIC   outputCol="features").transform(diamonds2) 
-- MAGIC 
-- MAGIC # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression
-- MAGIC from pyspark.ml.regression import LinearRegression
-- MAGIC lr2 = LinearRegression(maxIter=5, regParam=0.0, solver="normal", #weightCol="weight",
-- MAGIC                      featuresCol="features", labelCol="price")
-- MAGIC model2 = lr2.fit(assembler2)
-- MAGIC print("coef/inter", model2.coefficients, model2.intercept)
-- MAGIC 
-- MAGIC # Summarize the model over the training set and print out some metrics
-- MAGIC ts2 = model2.summary
-- MAGIC print("numIterations: %d" % ts2.totalIterations)
-- MAGIC print("objectiveHistory: %s" % str(ts2.objectiveHistory))
-- MAGIC #ts2.residuals.show()
-- MAGIC print("RMSE: %f" % ts2.rootMeanSquaredError)
-- MAGIC print("r2: %f" % ts2.r2)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC fig, ax = plt.subplots()
-- MAGIC ts2.residuals.sort("residuals").toPandas()["residuals"].plot()
-- MAGIC display(fig)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression
-- MAGIC from pyspark.ml.regression import DecisionTreeRegressor
-- MAGIC dtr = DecisionTreeRegressor(featuresCol="features", labelCol="price", maxDepth=15, maxBins=64)
-- MAGIC model3 = dtr.fit(assembler2)
-- MAGIC predictions = model3.transform(assembler2)
-- MAGIC 
-- MAGIC # Summarize the model over the training set and print out some metrics
-- MAGIC from pyspark.ml.evaluation import RegressionEvaluator
-- MAGIC evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
-- MAGIC rmse = evaluator.evaluate(predictions)
-- MAGIC print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC fig, ax = plt.subplots()
-- MAGIC predictions.withColumn("residuals", predictions.prediction - predictions.price).sort("residuals").toPandas()["residuals"].plot()
-- MAGIC display(fig)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(predictions.withColumn("residuals", predictions.prediction - predictions.price).describe())

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # use logs
-- MAGIC from pyspark.sql.functions import log
-- MAGIC diamonds3 = diamonds2.withColumn("logCarat", log(diamonds2.carat))
-- MAGIC diamonds3 = diamonds3.withColumn("logPrice", log(diamonds2.price))
-- MAGIC 
-- MAGIC display(diamonds3)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # https://www.instaclustr.com/support/documentation/apache-spark/spark-mllib-linear-regression-example/
-- MAGIC from pyspark.ml.feature import VectorAssembler
-- MAGIC 
-- MAGIC assembler3 = VectorAssembler(
-- MAGIC   inputCols=["logCarat", "depth", "table", "cutVec", "colorVec", "clarityVec", "x", "y", "z"], 
-- MAGIC   outputCol="features").transform(diamonds3) 
-- MAGIC 
-- MAGIC # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression
-- MAGIC from pyspark.ml.regression import LinearRegression
-- MAGIC lr3 = LinearRegression(maxIter=5, regParam=0.0, solver="normal", #weightCol="weight",
-- MAGIC                      featuresCol="features", labelCol="logPrice")
-- MAGIC                      #featuresCol="features", labelCol="price")
-- MAGIC model3 = lr3.fit(assembler3)
-- MAGIC print("coef/inter", model3.coefficients, model3.intercept)
-- MAGIC 
-- MAGIC # Summarize the model over the training set and print out some metrics
-- MAGIC ts3 = model3.summary
-- MAGIC print("numIterations: %d" % ts3.totalIterations)
-- MAGIC print("objectiveHistory: %s" % str(ts3.objectiveHistory))
-- MAGIC #ts2.residuals.show()
-- MAGIC print("RMSE: %f" % ts3.rootMeanSquaredError)
-- MAGIC print("r2: %f" % ts3.r2)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC predictions = model3.transform(assembler3)
-- MAGIC from pyspark.sql.functions import exp, pow, avg
-- MAGIC rmse = predictions.withColumn("sq_er", pow(exp(predictions.prediction) - predictions.price, 2))#.agg(avg("sq_er"))
-- MAGIC display(rmse)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(rmse.agg(avg("sq_er")))
-- MAGIC # pow(rmse.agg(avg("sq_er")), 0.5)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression
-- MAGIC from pyspark.ml.regression import DecisionTreeRegressor
-- MAGIC dtr2 = DecisionTreeRegressor(featuresCol="features", labelCol="logPrice", maxDepth=15, maxBins=64)
-- MAGIC model4 = dtr2.fit(assembler3)
-- MAGIC predictions4 = model4.transform(assembler3)
-- MAGIC rmse4 = predictions4.withColumn("er", exp(predictions4.prediction) - predictions4.price)#.agg(avg("sq_er"))
-- MAGIC rmse4 = rmse4.withColumn("sq_er", pow(rmse4["er"], 2))#.agg(avg("sq_er"))
-- MAGIC 
-- MAGIC 
-- MAGIC # Summarize the model over the training set and print out some metrics
-- MAGIC #from pyspark.ml.evaluation import RegressionEvaluator
-- MAGIC #evaluator = RegressionEvaluator(labelCol="logPrice", predictionCol="prediction", metricName="rmse")
-- MAGIC #rmse = evaluator.evaluate(predictions)
-- MAGIC #print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import sqrt
-- MAGIC #display(sqrt(rmse4.agg(avg("sq_er"))["avg(sq_er)"].head()[0]))
-- MAGIC display(rmse4.agg(avg("sq_er")).withColumn("rmse", sqrt("avg(sq_er)")))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(rmse4)

-- COMMAND ----------


