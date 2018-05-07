import pandas as pd
import tensorflow as tf
import numpy as np
#import input_data
from sklearn.utils import shuffle



df = pd.read_csv("data\\same_check.csv")
#df = shuffle(df) #打乱
df = df.drop("time", 1)

"""x_train = df.sample(frac=.95, random_state=1232)
x_test = df.drop(x_train.index)

x_train.to_csv("data\\x_train.csv", sep=',', encoding='utf-8', header=False, na_rep='', index=False)
x_test.to_csv("data\\x_test.csv", sep=',', encoding='utf-8', header=False, na_rep='', index=False)"""


#df = input_data.read_data()
_CSV_COLUMNS =df.columns.values.tolist()
_CSV_COLUMN_DEFAULTS=[]
			 	
for column in df.dtypes:
	if(column=="int64"):
		_CSV_COLUMN_DEFAULTS.append([0])
	elif(column=="float64"):
		_CSV_COLUMN_DEFAULTS.append([0.00])
	else:
		_CSV_COLUMN_DEFAULTS.append([''])



endsite = tf.feature_column.numeric_column ('endsite')
m = tf.feature_column.numeric_column ('m')
w = tf.feature_column.numeric_column ('w')
dept = tf.feature_column.numeric_column ('dept')


dept_buckets = tf.feature_column.bucketized_column(dept, boundaries=[5, 6, 7, 8, 10, 11])
#endsite_buckets = tf.feature_column.bucketized_column(endsite, boundaries=[260])
endsite_buckets = tf.feature_column.bucketized_column(endsite, boundaries=[4, 14, 17, 27, 28, 29, 32, 33, 37, 41, 44, 90, 94, 98, 100, 108, 141, 148, 150, 182, 184, 186, 196, 204, 210, 214, 228, 230, 231, 232, 241, 245, 252, 260, 261, 273, 274, 278, 285, 288, 290, 295,
																			   300, 303, 317, 326, 349, 356, 358, 363, 364, 376, 378, 391, 392, 393, 394, 395, 396, 397, 405, 407, 411, 434, 440, 445, 451, 452, 453, 482, 507, 526, 529, 530, 532, 534, 550, 551, 555, 590, 592, 
																			   596, 597, 612, 629, 632, 633, 634, 664, 677, 680, 683, 686, 700, 705, 707, 710, 714, 716, 722, 723, 725, 729, 750, 765, 798, 808, 811, 830, 843, 852, 853, 854, 865, 868, 874, 901, 904, 926, 927,
																			   931, 936, 948, 949, 969, 988, 996, 1001, 1006, 1011, 1030, 1031, 1032, 1041, 1053, 1055, 1064, 1075, 1076, 1078, 1091, 1093, 1113, 1119, 1120, 1123, 1130, 1135, 1137, 1144, 1152, 1161, 1167, 
																			   1178, 1181, 1191, 1194, 1195, 1212, 1219, 1223, 1230, 1233, 1237, 1247, 1248, 1253, 1259, 1260, 1271, 1292, 1294, 1297, 1306, 1307, 1310, 1320, 1384, 1461, 1472, 1537, 1558, 1559, 3600002, 4400002])
m_buckets = tf.feature_column.bucketized_column(m, boundaries=[1, 2, 3, 4, 5])
w_buckets = tf.feature_column.bucketized_column(w, boundaries=[1, 2, 3, 4, 5, 6, 7])



def input_fn(data_file, num_epochs, shuffle, batch_size):
	assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)
	def parse_csv(data_file):
		print('Parsing', data_file)
		columns = tf.decode_csv(data_file, record_defaults=_CSV_COLUMN_DEFAULTS)
		features = dict(zip(_CSV_COLUMNS, columns))
		labels = features.pop('num')
		return features, labels

	dataset = tf.data.TextLineDataset(data_file)
	dataset = dataset.map(parse_csv, num_parallel_calls=5)

	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	return features, labels


deep_columns = [dept_buckets, endsite_buckets, m_buckets, w_buckets]

model = tf.estimator.DNNRegressor(
#			 model_dir='tmp\\model_DNN',
			 feature_columns=deep_columns,
			 hidden_units=[256,128,64],
			 optimizer=tf.train.ProximalAdagradOptimizer(
			 	learning_rate=0.05,
			 	l1_regularization_strength=0.01),
			 config=tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0})))



model.train(input_fn=lambda: input_fn("data\\x_train.csv", 40, True, 10))

result = model.evaluate(input_fn=lambda: input_fn("data\\x_test.csv", 1, False, 10))

for key in sorted(result):
	print('%s: %s' % (key, result[key]))



#predictResult = model.predict(input_fn=lambda: input_fn("data\\x_test.csv", 1, False, 10))

#for i,j in enumerate(predictResult):
#	print("%s : %s" % (i+1, j))
	

