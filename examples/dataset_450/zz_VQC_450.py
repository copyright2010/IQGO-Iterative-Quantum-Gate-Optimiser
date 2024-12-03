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
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from IPython.display import clear_output
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, StatePreparation, PauliFeatureMap, StatePreparation



dataset = openml.datasets.get_dataset(450)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

data = pd.DataFrame(X)
data['y'] = pd.DataFrame(y)

data2 = data.dropna(axis=0)

X = np.array(data2)


def callback_graph(weights, obj_func_eval):

    clear_output(wait=False)
    objective_func_vals.append(obj_func_eval)
    #print(len(objective_func_vals))

objective_func_vals = []

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

print(df)

data_train2 = df.drop(columns=['labels'])

labels = df['labels']

scaler = MinMaxScaler()

print(Counter(labels))
from sklearn.model_selection import StratifiedKFold
from qiskit.circuit.library import RealAmplitudes
import tensorflow as tf

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)


seeds = np.array([42**4,231,42*4,123,3223,341,222])

acc_all = []
for noise in seeds:

    score, save, train_acc = [], [], []

    data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.66, random_state=noise, stratify = labels)
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


        num_qubits = 4
        backend = Aer.get_backend('statevector_simulator')
        seed = algorithm_globals.random_seed = 1234
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3,entanglement='full')

        #initializer = tf.keras.initializers.random_uniform(minval=0.15, maxval=0.85, seed= 111)         
        optimizer = COBYLA(maxiter=40)


        #initial_point = initializer(shape=(1,ansatz.num_parameters))

        zz_kernelf = ZZFeatureMap(num_qubits, reps=2, entanglement='linear')

        vqc = VQC(
            feature_map=zz_kernelf,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            callback=callback_graph,
            # initial_point=initial_point
        )

        from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

        encoder = OneHotEncoder()

        y_train_encoded = encoder.fit_transform(pd.DataFrame(y_train).values.reshape(-1, 1))
        y_test_encoded  = encoder.transform(pd.DataFrame(y_test).values.reshape(-1, 1))
        
        print("matrix_train_normalised shape:", matrix_train_normalised.shape)
        print("y_train shape:", y_train.shape)

    #    print(pd.DataFrame(y_train).values.reshape(-1, 1))

        fitted = vqc.fit(matrix_train_normalised, y_train_encoded)
        pred_train = fitted.predict(matrix_train_normalised)

        pred_train = encoder.inverse_transform(pred_train)

        scor_train = balanced_accuracy_score(y_train, pred_train)
        cm = confusion_matrix(y_train, pred_train)
        train_acc.append(scor_train)
        
        program = 'val'

        start_time = time.time()

        if program == 'test':

            end_time_train = time.time()
            execution_time = end_time_train - start_time
            print('Training Time: ',execution_time)

            pred = fitted.predict(matrix_test_normalised)
            
            end_time_predict = time.time()
            prediction_time = end_time_predict - end_time_train
            print('execution Time: ', prediction_time)
        
            pred = encoder.inverse_transform(pred)

            scor = balanced_accuracy_score(y_test, pred)
            cm = confusion_matrix(y_test, pred)

        elif program == 'val':

            end_time_train = time.time()
            execution_time = end_time_train - start_time
            print('Training Time: ',execution_time)

            pred = fitted.predict(matrix_val_normalised)
            
            end_time_predict = time.time()
            prediction_time = end_time_predict - end_time_train
            print('execution Time: ', prediction_time)
            pred = encoder.inverse_transform(pred)

            scor = balanced_accuracy_score(val_labels, pred)
            cm = confusion_matrix(val_labels, pred)

        print(scor)
        print(cm)

        save_score = [scor]

        score.append(save_score)

    acc_all.append(np.mean(score))

print('Train acc: ', np.mean(train_acc))

savedf = pd.DataFrame(acc_all)

print('Acc mean val: ',np.mean(savedf))

print(savedf)
# print(savedf)

    #savedf.to_csv('tabnet_test_acc_11.csv')
