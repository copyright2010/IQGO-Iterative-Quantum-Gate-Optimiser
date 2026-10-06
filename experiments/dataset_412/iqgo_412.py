import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from collections import Counter
import numpy as np
# import openml
from IQGO_module.iqgo_training import IQGO_train
import torch.nn as nn

# dataset = openml.datasets.get_dataset(450)
# X, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format="array", target=dataset.default_target_attribute)


data_frame = pd.read_csv('412_fig6_r2_param_estimates.csv', sep=",", index_col=None).dropna(axis=1)
# data_frame = pd.read_csv('new_df_param_estimates_new.csv', sep=",", index_col=None).dropna(axis=1)

X = data_frame[['0.1','0.2','0.3']]
y = data_frame['0.4']

data = pd.DataFrame(X)
data['y'] = pd.DataFrame(y)

data2 = data.dropna(axis=0)

X = np.array(data2)

data_train = pd.DataFrame()
data_train['0.1'] = (X[:,0])
data_train['0.2'] = (X[:,1])
data_train['0.3'] = (X[:,2])

df = pd.DataFrame()

df['0.1'] = (data_train['0.1'])
df['0.2'] =  data_train['0.2']
df['0.3'] = (data_train['0.3'])
df['labels'] = y

print(Counter(df['labels']),'................all labels..................')

data_train2 = df.drop(columns=['labels'])
labels2 = df['labels']

data_train2 = data_train2[25:]#.dropna()
labels = labels2[25:]

#seeds = np.array([4**12,231,12314,345,353,6,34553,3646,5634,2342,234,546,564])

acc_test, acc_val, save_noise, cct, acc_train, metrics_all = [], [], [], [], [], []

# seeds = np.array([42**4])

# seeds = np.array([5634,2342,234,546,564])
# seeds = np.array([564])

# seeds = np.array([42**4,231,42*4,123,3223,341,222])
seeds = np.array([4*12,231*5,42**4])
# seeds = np.array([42**4,231,42*4,123,3223])
# seeds = np.array([450*1,450*2,450*3,450*4,450*5])

# seeds = np.array([4*12,231*5,42**4])
 ### bad ones
seeds = np.array([231*5,626,5451,123,4332,234,4,4*12,1233,343])*1#  #]

for noise in seeds:

    init_iqgo = IQGO_train(noise_level = 0.0, seed_val=noise, kfold_splits=5)
    
    fitted_circuit = init_iqgo.fit(data_train=data_train2, labels=labels, number_of_layers = 4)

    # fitted_circuit = np.array([8, 6, 3]) # best
    
    predictions, column_means = init_iqgo.predict(data_train = data_train2, labels = labels, compiled_circuit=fitted_circuit,mode='test')
    acc_test.append(column_means)
    
    # predictions, column_means = init_iqgo.predict(data_train = data_train2, labels = labels, compiled_circuit=fitted_circuit,mode='val')
    predictions, column_means, metrics_df = init_iqgo.predict_validate(data_train = data_train2, labels = labels, compiled_circuit=fitted_circuit)
    metrics_df["seed"] = noise

    acc_val.append(column_means)
    
    save_noise.append(noise)
    metrics_all.append(metrics_df)
    # acc_train.append(train_acc)
    cct.append(fitted_circuit)
    print('done')
    print(fitted_circuit)



print('Acc test: ',acc_test)
print('Acc val: ',acc_val)

metrics_seed_df = pd.DataFrame(metrics_all)

print(metrics_seed_df)

mean_metrics = metrics_seed_df.drop(columns=["seed"]).mean()
std_metrics = metrics_seed_df.drop(columns=["seed"]).std()

print(mean_metrics)
print(std_metrics)

# print('Acc mean train: ',np.mean(acc_train))
print('Acc mean test: ',np.mean(acc_test))
print('Acc mean val: ',np.mean(acc_val))


save_noise = np.transpose(save_noise)

print(save_noise)
all = pd.DataFrame([save_noise, acc_test, acc_val, cct])

all.to_csv('results_new_iqgo.csv')
