import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from collections import Counter
import numpy as np
import openml
from IQGO_module.iqgo_training import IQGO_train

dataset = openml.datasets.get_dataset(450)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

data = pd.DataFrame(X)
data['y'] = pd.DataFrame(y)

data2 = data.dropna(axis=0)

X = np.array(data2)

data_train = pd.DataFrame()
data_train['0.1'] = (X[:,0])
data_train['0.2'] = (X[:,1])
data_train['0.3'] = (X[:,2])
data_train['0.4'] = (X[:,3])

df = pd.DataFrame()

df['0.1'] = np.log(data_train['0.1'])
df['0.2'] = data_train['0.2']
df['0.3'] = data_train['0.3']
df['0.4'] = data_train['0.4']
df['labels'] = X[:,4]

print(Counter(df['labels']),'................all labels..................')

data_train2 = df.drop(columns=['labels'])
labels = df['labels']

#seeds = np.array([4**12,231,12314,345,353,6,34553,3646,5634,2342,234,546,564])

acc_test, acc_val, save_noise, cct, acc_train = [], [], [], [], []

# seeds = np.array([42**4])

# seeds = np.array([5634,2342,234,546,564])
# seeds = np.array([564])

seeds = np.array([42**4,231,42*4,123,3223,341,222])

for noise in seeds:

    init_iqgo = IQGO_train(noise_level = 0.0, seed_val=noise, kfold_splits=5)
    
    fitted_circuit = init_iqgo.fit(data_train=data_train2, labels=labels, number_of_layers = 2)

    # fitted_circuit = np.array([8, 6, 3]) # best
    
    predictions, column_means, train_acc = init_iqgo.predict(data_train = data_train2, labels = labels, compiled_circuit=fitted_circuit,mode='test')
    acc_test.append(column_means)
    
    predictions, column_means, train_acc = init_iqgo.predict(data_train = data_train2, labels = labels, compiled_circuit=fitted_circuit,mode='val')
    
    acc_val.append(column_means)
    save_noise.append(noise)
    acc_train.append(train_acc)
    cct.append(fitted_circuit)
    print('done')
    print(fitted_circuit)


print('Acc test: ',acc_test)
print('Acc val: ',acc_val)

print('Acc mean test: ',np.mean(acc_train))
print('Acc mean test: ',np.mean(acc_test))
print('Acc mean val: ',np.mean(acc_val))


save_noise = np.transpose(save_noise)

print(save_noise)
all = pd.DataFrame([save_noise, acc_test, acc_val, cct])

all.to_csv('results3.csv')