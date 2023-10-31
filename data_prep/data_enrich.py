
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
spark.sparkContext._conf.setAll([('spark.sql.files.maxPartitionBytes', '500mb'), ("spark.sql.shuffle.partitions", 16)])
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import os
from pyspark.sql.window import Window
from pyspark.sql.functions import collect_list
from datetime import datetime
def convert_to_timestamp(value):
    # if value >= 240000:
    #     hours = value // 10000 
    #     minutes = (value // 100) % 100
    #     seconds = value % 100
    # else:
    hours = value // 10000
    minutes = (value // 100) % 100
    seconds = value % 100
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

convert_to_timestamp_udf = udf(convert_to_timestamp, StringType())
# Params
start_date = '2023-01-01'
start_dttm = '2023-01-01 00:00:00'

end_date = '2023-08-01'
end_dttm = '2023-08-01 00:00:00'

order_start_date = '2022-12-19'

# Paths
schema = 's3://'

input_bucket_name = 'qzhao-promise'
input_project_name = 'ML_feature'
input_folder = 'transaction_inventory_fc_distance_rank_shipment_route_file_final_v0'

output_bucket_name = 'ogunes-promise'
output_project_name = 'EDD'
output_folder = 'reporting/data'

PDD_input_data_path = os.path.join(
    schema, 
    input_bucket_name, 
    input_project_name, 
    input_folder)
tableau_data_temp_path = os.path.join(
    schema, 
    output_bucket_name, 
    output_project_name, 
    output_folder, 
    '2023-10-24_vEST')
tableau_data_path = os.path.join(
    schema, 
    output_bucket_name, 
    output_project_name, 
    output_folder, 
    '2023-10-25_vEST_vReleaseCutOff')
data = spark.read.parquet(PDD_input_data_path)
data.printSchema()
data = data.withColumn("delivery_day_of_week", 
                       F.dayofweek(data["SHIPMENT_DELIVERY_DATE"]) - 1)
feature_set = ['ORDER_ID',
               'CUSTOMER_POSTCODE',
               'ORDER_PLACED_DTTM_EST',
               'RELEASE_DTTM_EST',
               'PRODUCT_KEY',
               'FFMCENTER_NAME',
               'location_code',
               'SHIPMENT_DELIVERY_DATE_EST',
               'SHIPMENT_ESTIMATED_DELIVERY_DATE',
               'delivery_day_of_week',
               'UOM_flag',
               'arc_range',
               'distance_mi',
               'Zone',
               'RouteID',
               'ACTUAL_TRANSIT_DAYS',
               'SHIPMENT_SHIPPED_DTTM_EST',
               'WAREHOUSE_ACTUAL_SHIP_DTTM_EST',
               'ACTUAL_CUTOFF_TIME']
data = data.select(feature_set)
data = data.filter(F.col('location_code') == F.col('FFMCENTER_NAME'))
data = data.dropDuplicates()
tableau_data_temp_path
data.repartition(20).write.parquet(tableau_data_temp_path)
data = spark.read.parquet(tableau_data_temp_path)
data.count()
sr_edd = glueContext.create_dynamic_frame.from_catalog(
    database='shipment_inspecter_table_promise', 
    table_name='shipment_route_EDD_history')
sr_edd = sr_edd.toDF()
# DUPLICATES !!!
# sr_edd.filter( (F.col('DATE') == '2023-02-05') & (F.col('FCName') == 'DFW1')  & (F.col('Zip5') == '80524') ).show()
core_fc = ['AVP1','AVP2','CFC1','CLT1',
           'DAY1','DFW1','EFC3','MCI1',
           'MCO1','MDT1','PHX1','RNO1',
           'WFC2', 'BNA1']
sr_edd = sr_edd.filter(
    (F.col('DATE') >= order_start_date)
    &(F.col('DATE') <= end_date)
    & (F.col('MODE').isin(['FDXHD']))
    & F.col('FCName').isin(core_fc)).cache()


window_spec = Window.partitionBy(
    ['DATE', 'FCName' ,'RouteID', 'Zip5']).orderBy(
    F.col("AdjTNT"))
sr_edd = sr_edd.withColumn(
    "row_number", 
    F.row_number().over(window_spec))           

sr_edd = sr_edd.filter(
    F.col('row_number') == 1).drop('row_number')  

sr_edd = sr_edd.select(F.col('FCName').alias('location_code'),
                       F.col('Zip5').alias('CUSTOMER_POSTCODE'),
                       F.col('DATE').alias('RELEASE_DATE_EST'),
                       F.col('Cutoff').alias('Cutoff_release'),
                       F.col('RouteID').alias('RouteID_release'),
                       F.col('AdjTNT').alias('AdjTNT_release'),
                       F.col('NextAdjTNT').alias('NextAdjTNT_release')).dropDuplicates()
data = data.withColumn("RELEASE_DATE_EST", 
                       F.substring(F.col("RELEASE_DTTM_EST"), 1, 10))

data = data.join(sr_edd,
                on=['RELEASE_DATE_EST', 'location_code', 'CUSTOMER_POSTCODE'],
                how='left')

data = data.filter((F.col('Cutoff_release').isNotNull())).dropDuplicates()
data = data.withColumn("Cutoff_release_stamp", convert_to_timestamp_udf("Cutoff_release"))
data.printSchema()
tableau_data_path
data.repartition(20).write.parquet(tableau_data_path)
out = spark.read.parquet(tableau_data_path)
out.count()
out.count()
out.printSchema()
# col rename and selection
out = out.select(F.col('ORDER_ID').alias('order_id'),
                F.col('UOM_flag'),
                F.col('FFMCENTER_NAME').alias('fc_name'),
                F.col('SHIPMENT_DELIVERY_DATE_EST').alias('shipment_delivery_date'),
                F.col('SHIPMENT_SHIPPED_DTTM_EST').alias('shipment_shipped_dttm'),
                F.col('Cutoff_release_stamp').alias('cutoff'), 
                F.col('RouteID_release').alias('route_id'), 
                F.col('RELEASE_DTTM_EST').alias('release_dttm'),
                F.col('delivery_day_of_week').alias('delivery_dow'),
                F.col('arc_range'),
                F.col('distance_mi'),
                F.col('Zone').alias('zone'),
                F.col('CUSTOMER_POSTCODE').alias('customer_zip'),
                F.col('SHIPMENT_ESTIMATED_DELIVERY_DATE').alias('EDD_v0'),
                F.col('AdjTNT_release').alias('adj_TNT'),
                F.col('NextAdjTNT_release').alias('nextadj_tnt'))
# STD estimation v0
out = out.withColumn('std_v0', 
                     F.datediff(F.col("EDD_v0"),
                                F.col("shipment_shipped_dttm") 
                                ))
out.show(5)
# order release dow
out = out.withColumn("release_dow", F.dayofweek(out["release_dttm"]) - 1)

# fc shipping scan dow
out = out.withColumn("ship_dow", F.dayofweek(out["shipment_shipped_dttm"]) - 1)
# STD day count
out = out.withColumn("std_actual", 
                     F.datediff(F.col("shipment_delivery_date"), 
                                F.col("shipment_shipped_dttm")))
out.show()
out.groupBy('std_actual').count().orderBy(F.col('count').desc()).show()
# fc type
G2 = ['AVP2', 'MCI1', 'RNO1', 'BNA1']
# after cpt
out = out.withColumn(
    'CPT',
    F.when(F.col('fc_name').isin(G2), 
      F.col('cutoff') + F.expr('INTERVAL 5 HOURS')
      ).otherwise( 
    F.col('cutoff') + F.expr('INTERVAL 3 HOURS') ))

out = out.withColumn("CPT", 
                     F.substring(F.col("CPT"), 12, 8))

out = out.withColumn("CPT", 
                     F.date_format(F.to_timestamp(F.col("CPT"), 'HH:mm:ss'), "HH:mm:ss"))

out = out.withColumn("shipment_shipped_timestamp", 
                     F.substring(F.col("shipment_shipped_dttm"), 12, 8))

out = out.withColumn("shipment_shipped_timestamp", 
                     F.date_format(F.to_timestamp(F.col("shipment_shipped_timestamp"), 'HH:mm:ss'), "HH:mm:ss"))
out = out.withColumn('is_after_CPT',
                    F.when(F.col('shipment_shipped_timestamp') > F.col('CPT'),
                          F.lit(1)).otherwise(F.lit(0)))

out = out.withColumn('is_after_CPT',
                    F.when( (F.col('CPT') < '04:00:00') & (F.col('shipment_shipped_timestamp') > '04:00:00'),
                          F.lit(0)).otherwise(F.col('is_after_CPT')))
# is holiday
holiday_path = os.path.join(
    schema, 
    output_bucket_name, 
    output_project_name, 
    'holiday',
    'FdxHolidaysWeb.csv')
holiday = spark.read.csv(holiday_path, header=True)
def user_defined_timestamp(date_col):
    _date = datetime.strptime(date_col, '%m/%d/%y')
    return _date.strftime('%Y-%m-%d')
user_defined_timestamp_udf = F.udf(user_defined_timestamp, StringType())
holiday = holiday.withColumn("Date", user_defined_timestamp_udf('Date'))
out = out.withColumn("shipment_shipped_date", 
                     F.substring(F.col("shipment_shipped_dttm"), 1, 10))
out = out.join(holiday.select(F.col('Date').alias('shipment_shipped_date'),
                       F.col('Holiday'), F.col('Note')),
        'shipment_shipped_date',
        'left')

# out.filter(F.col('Note').isNotNull()).show()
out = out.withColumn("holiday_flag", 
                     F.when(
                         F.col('Holiday').isNotNull(), 
                         F.lit('holiday')).otherwise(F.lit('regular')))
    
out = out.withColumn("holiday_flag", 
                     F.when(
                         F.col('Note').isNotNull(), 
                         F.lit('before_holiday')).otherwise(F.col('holiday_flag')))
# out.filter(F.col('holiday_flag') != 'regular').show()
# is friday or saturday
out = out.withColumn("is_release_fri_sat", 
                     F.when(
                         F.col('release_dow').isin([5,6]), F.lit(1)).otherwise(F.lit(0))
                     )

out = out.withColumn("is_ship_fri_sat", 
                     F.when(
                         F.col('ship_dow').isin([5,6]), F.lit(1)).otherwise(F.lit(0))
                     )
out = out.withColumn("release_fri_sat_tag", 
                     F.when(
                         F.col('release_dow').isin([5]), F.lit('friday')).otherwise(
                         F.when(F.col('release_dow').isin([6]), F.lit('saturday')).otherwise(
                         F.lit('regular')))
                     )

out = out.withColumn("ship_fri_sat_tag", 
                     F.when(
                         F.col('ship_dow').isin([5]), F.lit('friday')).otherwise(
                         F.when(F.col('ship_dow').isin([6]), F.lit('saturday')).otherwise(
                         F.lit('regular')))
                     )
out.show()
# time till next CPT
out = out.withColumn("CPT",F.to_timestamp(F.col("CPT"),"HH:mm:ss")) \
   .withColumn("shipment_shipped_timestamp",F.to_timestamp(F.col("shipment_shipped_timestamp"),"HH:mm:ss"))
out = out.withColumn("CPT", 
                     F.when(
                         F.col('CPT').isNull(), F.to_timestamp(F.lit("00:00:00"),"HH:mm:ss")).otherwise(F.col('CPT'))
                    )
out = out.withColumn("hours_till_CPT", 
                     F.round((F.col("CPT").cast("long")-F.col("shipment_shipped_timestamp").cast("long"))/3600,2))
out = out.withColumn("hours_till_CPT", 
                     F.when(
                         F.col('hours_till_CPT')<0, F.col('hours_till_CPT')+24).otherwise(F.col('hours_till_CPT'))
                     )
# + F.expr('INTERVAL 3 HOURS')
out = out.withColumn("hours_till_CPT", 
                     F.round(F.col('hours_till_CPT'), 2))
out.printSchema()
col_target = ['delivery_dow', 'std_actual']
col_order = ['order_id', 'route_id', 'fc_name', 'UOM_flag', 'customer_zip', 
             'shipment_delivery_date', 'holiday_flag', 
             'is_release_fri_sat', 'is_ship_fri_sat', 'release_fri_sat_tag', 'ship_fri_sat_tag',
             'release_dow', 'ship_dow',
             'is_after_CPT', 'hours_till_CPT', 
             'adj_TNT', 'nextadj_tnt', 'std_v0']
col_route = ['arc_range', 
             'distance_mi', 
             'zone']

df_train_val_test = out.select(col_order + col_route + col_target)
df_dayofweek = df_train_val_test.select('release_dow').distinct().toPandas()
dayofweek_list = sorted(list(df_dayofweek['release_dow']))
dayofweek_dict = {value: index for index, value in enumerate(dayofweek_list)}
dayofweek_dict
def onehotencoding(feature, feature_dict, df):
    for k, v in feature_dict.items():
        df = df.withColumn(feature+'_'+ str(k), F.when(F.col(feature) == v, 1).otherwise(0))
    return df

df_train_val_test = onehotencoding('release_dow', dayofweek_dict, df_train_val_test)
df_train_val_test = onehotencoding('ship_dow', dayofweek_dict, df_train_val_test)
target_col = col_target
feature_cols = col_route + ['UOM_flag', 'holiday_flag', 
                            'is_release_fri_sat', 'is_ship_fri_sat', 'release_fri_sat_tag', 'ship_fri_sat_tag', 
                            'hours_till_CPT', 'is_after_CPT',
                            'adj_TNT', 'nextadj_tnt', 'std_v0'] + \
                            ['release_dow_0', 'release_dow_1', 'release_dow_2',
                            'release_dow_3', 'release_dow_4', 'release_dow_5',
                            'release_dow_6'] + \
                            ['ship_dow_0', 'ship_dow_1', 'ship_dow_2',
                            'ship_dow_3', 'ship_dow_4', 'ship_dow_5',
                            'ship_dow_6'] + \
                            ['order_id', 'fc_name', 'customer_zip', 'route_id']
print(len(feature_cols), 'feaures are generated')
# Dates
train_start_date = '2023-01-01'
train_end_date = '2023-05-01'

val_start_date = '2023-05-01'
val_end_date = '2023-06-01'

test_start_date = '2023-06-01'
test_end_date = '2023-08-01'

exp_date_string = '20231025_v8'
# Output Paths

output_folder_path = os.path.join(schema,
                                  output_bucket_name,
                                  output_project_name,
                                  'ml_train_val_test_set', 
                                  exp_date_string)

training_validation_path = os.path.join(output_folder_path,
                                        f'train_val_{train_start_date[:10]}_{val_end_date[:10]}')
print(training_validation_path)


training_path = os.path.join(output_folder_path,
                             f'train_{train_start_date[:10]}_{train_end_date[:10]}')
print(training_path)


validation_path = os.path.join(output_folder_path,
                               f'val_{val_start_date[:10]}_{val_end_date[:10]}')
print(validation_path)


test_path = os.path.join(output_folder_path,
                               f'test_{test_start_date[:10]}_{test_end_date[:10]}')
print(test_path)
df_train_val = df_train_val_test\
    .filter((F.col('shipment_delivery_date') < test_start_date))\
    .select(target_col + feature_cols)
df_train_val.write.mode('overwrite').parquet(training_validation_path)

df_train = df_train_val_test\
    .filter((F.col('shipment_delivery_date') < val_start_date))\
    .select(target_col + feature_cols)
df_train.write.mode('overwrite').parquet(training_path)

df_val = df_train_val_test\
    .filter(
        (F.col('shipment_delivery_date') < test_start_date) & 
        (F.col('shipment_delivery_date') >= val_start_date))\
    .select(target_col + feature_cols)
df_val.write.mode('overwrite').parquet(validation_path)

df_test = df_train_val_test\
    .filter(
        (F.col('shipment_delivery_date') >= test_start_date) & 
        (F.col('shipment_delivery_date') < test_end_date))\
    .select(target_col + feature_cols)
df_test.write.mode('overwrite').parquet(test_path)
df_train_val_test.count()
df_train_val_test.count()
df_train_val_test.show()
out.count()
out[['order_id']].dropDuplicates().count()
job.commit()