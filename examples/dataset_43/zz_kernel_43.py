import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from collections import Counter
import openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from sklearn.svm import SVC


dataset = openml.datasets.get_dataset(43)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

data = pd.DataFrame(X)
data['y'] = pd.DataFrame(y)

data2 = data.dropna(axis=0)
X = np.array(data2)

data_train = pd.DataFrame()
data_train['0.1'] = (X[:,0])
data_train['0.2'] = (X[:,1])
data_train['0.3'] = (X[:,2])

df = pd.DataFrame()
df['0.1'] = np.log(data_train['0.1'])
df['0.2'] = data_train['0.2']
df['0.3'] = data_train['0.3']
df['labels'] = X[:,3]

data_train2 = df.drop(columns=['labels'])
labels = df['labels']

print(Counter(labels))

svc = SVC(kernel='rbf')

scaler = MinMaxScaler()
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)

save, score = [], []

data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.66, random_state=123, stratify = labels)
rus = RandomUnderSampler(random_state=42)

data_train, train_labels = rus.fit_resample(data_train, train_labels)

for train_index, test_index in kf.split(data_train, train_labels):

    X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

    print(Counter(y_train))
    print(Counter(y_test))
    print(Counter(val_labels))

    matrix_train_normalised = scaler.fit_transform(X_train)
    
    matrix_test_normalised = scaler.transform(X_test)

    matrix_val_normalised = scaler.transform(data_val)  

    num_qubits = X_train.shape[1]

    from sklearn.svm import SVC
    from qiskit.circuit.library import ZZFeatureMap

    zz_kernelf = ZZFeatureMap(num_qubits, reps=3, entanglement='circular')

    zz_kernel = QuantumKernel(feature_map=zz_kernelf, quantum_instance=Aer.get_backend('statevector_simulator'))

    matrix_train = zz_kernel.evaluate(matrix_train_normalised)

    svc = SVC(kernel='precomputed')

    fitted = svc.fit(matrix_train, y_train)

    program = 'test'

    start_time = time.time()

    if program == 'test':

        end_time_train = time.time()
        execution_time = end_time_train - start_time
        print('Training Time: ',execution_time)
        matrix_test = zz_kernel.evaluate(x_vec = matrix_test_normalised,y_vec=matrix_train_normalised)

        pred = fitted.predict(matrix_test)
        
        end_time_predict = time.time()
        prediction_time = end_time_predict - end_time_train
        print('execution Time: ', prediction_time)
    
        scor = balanced_accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)

    elif program == 'val':

        end_time_train = time.time()
        execution_time = end_time_train - start_time
        print('Training Time: ',execution_time)
        matrix_val= zz_kernel.evaluate(x_vec = matrix_val_normalised,y_vec=matrix_train_normalised)

        pred = fitted.predict(matrix_val)
        
        end_time_predict = time.time()
        prediction_time = end_time_predict - end_time_train
        print('execution Time: ', prediction_time)
    
        scor = balanced_accuracy_score(val_labels, pred)
        cm = confusion_matrix(val_labels, pred)

    print(scor)
    print(cm)

    save_score = [scor]

    score.append(save_score)

savedf = pd.DataFrame(score)

print(savedf)
print(savedf.mean())

#savedf.to_csv('save_results_qkernel.csv')
