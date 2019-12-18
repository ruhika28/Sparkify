# Databricks notebook source
'''packages that need to be imported'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql.functions import concat, lit, avg, split, isnan, when, count, col, min, max, round
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import FloatType, IntegerType

'''read the data'''
df = spark.read.json('dbfs:/FileStore/tables/mini_sparkify_event_data.json')
cnt = df.count
print(cnt)
def clean_sparkify(df):
    '''removing the data  that has the null user id '''
    df_new = df.filter(df["userId"] != "")
    
    return df_new

df = clean_sparkify(df)
df.show(5)
df.persist()

df_final = df.where(df.firstName.isNotNull())
df_final.count()
df_final.head()


''' users who have confirmed their  cancellation but have valid  user  ids and and  not null'''
df_c = df_final.where(df_final.page == "Cancellation Confirmation") \
               .select("userId") \
               .distinct()

ids_c = [user_log['userId'] for user_log in df_c.collect()]
df_c = df_final.withColumn("isChurn", df_final.userId.isin(ids_c))
df_c.persist()
df_c.show()


# COMMAND ----------

df_c.head()
df_c.count()
df_c.show(5)

churn_df = df_c.select("userId", "isChurn") \
               .dropDuplicates() \
               .orderBy("userId")
# changing data type to int
churn_df = churn_df.select("userId", churn_df["isChurn"].cast("int"))

churn_df.persist()

df_c = df_final.withColumn("isChurn", df_final.userId.isin(ids_c))
df_c.head()







# COMMAND ----------

''' Scouting out the important  features for modelling'''

''' gender'''

gender_frame = df_c.select(["userId", "gender"]) \
                .dropDuplicates()

genders = gender_frame.select("gender") \
                   .distinct() \
                   .rdd \
                   .flatMap(lambda x: x) \
                   .collect()

genders_expr = [when(col("gender") == g, 1) \
                 .otherwise(0) \
                 .alias("gender" + g.capitalize()) \
                for g in genders]

gender_frame = gender_frame.select("userId", "gender", *genders_expr) \
                     .drop("gender") \
                     .orderBy("userId")

gender_frame.show(5)


''' songs browsed per user '''


song_frame = df_c.filter(df_c.page == "NextSong") \
              .select(["userId"]) \
              .dropDuplicates() \
              .groupby("userID") \
              .count()

song_frame = song_frame.withColumnRenamed("count", "songCount") \
                 .orderBy("userId")

song_frame.show(5)

''' number of sessions per user '''


session_frame = df_c.select(["userId", "sessionId"]) \
                 .dropDuplicates() \
                 .groupby("userId") \
                 .count()

session_frame = session_frame.withColumnRenamed("count", "sessionCount") \
                       .orderBy("userId")

session_frame.show(5)



''' songs per  session '''

avg_songs_per_session_df = df_c.filter(df_c.page =="NextSong") \
                               .groupBy(["userId", "sessionId"]) \
                               .count()

avg_songs_per_session_df = avg_songs_per_session_df.groupby("userId") \
                          .agg(avg(avg_songs_per_session_df["count"]) \
                          .alias("avgSongsPerSession")) \
                          .orderBy("userId")

avg_songs_per_session_df.show(5)


''' time consumed per session'''
avg_time_per_session_df = df_c.groupby("userId", "sessionId") \
                              .agg(((max(df_c.ts)-min(df_c.ts))/(1000*60)) \
                              .alias("sessionDurationMinutes"))
avg_time_per_session_df = df_c.groupby("userId", "sessionId") \
                              .agg(((max(df_c.ts)-min(df_c.ts))/(1000*60)) \
                              .alias("sessionDurationMinutes"))


avg_time_per_session_df.show(6)





''' no of artists '''

artist_df = df_c.filter(df.page == "NextSong") \
                 .select("userId", "artist") \
                 .dropDuplicates() \
                 .groupBy("userId") \
                 .count() \
                 .orderBy("userId")

artist_df = artist_df.withColumnRenamed("count", "artistCount")

artist_df.show(5)

# COMMAND ----------

  engd_features = [gender_frame, song_frame, session_frame, avg_songs_per_session_df, \
                 avg_time_per_session_df, \
                 artist_df, churn_df]

master = df_c.select("userID").dropDuplicates()
master.show(5)

# COMMAND ----------

''' merging the feature engineered vectorss'''

for f in engd_features:
   
    f = f.withColumnRenamed("userId", "userIdDrop")
    master = master.join(f, on = master["userId"] == f["userIdDrop"], how = 'left').drop("userIdDrop")


# COMMAND ----------

master.show(5)

# COMMAND ----------

assembled_vector = VectorAssembler(inputCols = master.columns[1:-1],
                            outputCol = "Feature_vector")

data_set = assembled_vector.transform(master)


'''Standardize the result'''

scaled = StandardScaler(inputCol = "Feature_vector",
                        outputCol = "FeaturesScaled",
                        withStd = True)
scaler = scaled.fit(data_set)
data_set_final = scaler.transform(data_set)



data_set_final = data_set_final.select(data_set_final.isChurn.alias("label"),
                   data_set_final.FeaturesScaled.alias("features"))

data_set_final.show(5)


''' 80 20 random split'''
training, test = data_set_final.randomSplit([0.8, 0.2],
                              seed = 42)

training.cache()

# COMMAND ----------

training.show(5)
test.show(5)

# COMMAND ----------

lr = LogisticRegression()

''' elastic net has been used in order to aviod overfitting '''


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(training)

# COMMAND ----------

print(lr_model)
print(lr)

# COMMAND ----------

paramGrid = ParamGridBuilder() \
                .addGrid(lr.elasticNetParam, [0.0, 0.4, 0.8]) \
                .addGrid(lr.regParam, [0.0, 0.8]) \
                .addGrid(lr.maxIter, [5]) \
                .build()


cv = CrossValidator(estimator = lr,
                           estimatorParamMaps = paramGrid,
                           evaluator = MulticlassClassificationEvaluator(),
                           numFolds = 2)
lrModel = cv.fit(training)

print(lrModel)
''' 71 percent accuracy'''

# COMMAND ----------

''' random forest '''

rf = RandomForestClassifier()
paramGrid = ParamGridBuilder() \
            .addGrid(rf.impurity,['entropy', 'gini']) \
            .addGrid(rf.maxDepth,[5, 10]) \
            .addGrid(rf.seed, [42]).build()

cv = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=MulticlassClassificationEvaluator(),
                    numFolds=3)

cv_rf = cv.fit(training)

cv_rf.save("cv_rf.model")

cv_rf.avgMetrics
''' 78 percent accuracy'''   
