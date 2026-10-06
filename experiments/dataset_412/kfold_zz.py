# test_script.py
# from IQGO_module.iqgo_training import IQGO_train
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier
# import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
# from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
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
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile
from scipy.stats import levy_stable

def add_levy_noise(X, mean=0, alpha=2.0,gamma=0.071, seed=42):
    np.random.seed(seed)  # Set the seed
    X_noisy = X.copy()
    alpha = alpha  # Characteristic exponent (0 < alpha <= 2)
    beta = 0     # Symmetry parameter (-1 ≤ beta ≤ 1)
    gamma = gamma    # Scale parameter (gamma > 0)
    delta = 0    # Location parameter

    # Generate α-stable noise
    size = X_noisy.shape[0]  # Number of samples

    for column in X_noisy.columns:
        noise = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=size)

        # rolling_std = X_noisy[column].std()
        # gaussian_noise = np.random.normal(mean, rolling_std*std, X[column].shape)
        X_noisy[column] += noise

    return X_noisy

def compute_metrics(y_true, y_pred):
    metrics = {"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        # Good default for imbalanced or multiclass classification
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        # Weighted by class support
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    return metrics


# data_frame = pd.read_csv('new_df_param_estimates_new.csv', sep=",").dropna(axis=1)
data_frame = pd.read_csv('412_fig6_r2_param_estimates.csv', sep=",").dropna(axis=1)

#data_frame = pd.read_csv('MM3.csv', sep=",").dropna(axis=1)
indx = data_frame.index
column = data_frame.columns
data_train = data_frame.drop(columns=['0.4'])

labels2 = data_frame['0.4']
# data_train = data_frame.drop(columns=['0.4'])
# labels = data_frame['0.4']
df = pd.DataFrame()

df['0.1'] = data_train['0.1']
df['0.2'] = data_train['0.2']
df['0.3'] = data_train['0.3']

# df['0.4'] = data_train['0.1'].shift(5)
# df['0.5'] = data_train['0.2'].shift(5)
# df['0.6'] = data_train['0.3'].shift(5)

# df['0.7'] = data_train['0.1'].shift(10)
# df['0.8'] = data_train['0.2'].shift(10)
# df['0.9'] = data_train['0.3'].shift(10)

# df['1.3'] = data_train['0.1'].shift(20)
# df['1.4'] = data_train['0.2'].shift(20)
# df['1.5'] = data_train['0.3'].shift(20)

# df['1.0'] = data_train['0.1'].shift(25)
# df['1.1'] = data_train['0.2'].shift(25)
# df['1.2'] = data_train['0.3'].shift(25)

data_train2 = df[25:]#.dropna()
labels = labels2[25:]
##


print(Counter(labels))

svc = SVC(kernel='rbf')

scaler = MinMaxScaler()
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=128*1)


# seeds = np.array([12*10,12*15,123*25,12*35,12*45,12*55,12*65,12+75])*2
# seeds = np.array([12*10])
# seeds = np.array([12*10,12*15,123*25,12*35,12*45])*2
seeds = np.array([12*10,12*15])*2 #,123*25,12*35,12*45
seeds = np.array([12*10,12*15,123*25,12*35,12*45])*2
seeds = np.array([44,626])#,52526,89741,1546,16885,84964,16548,4894561,654,4894,65,64,98774,654,984,1651,984,98456])*1#099
seeds = np.array([44,626,5451,123,4332,4255,23,234,5,4])*1#

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# Create quantum circuits for generating Block-Encoded State (BPS) wavefunctions of all data pairs 
# def make_bsp(n_dim):
#     param = ParameterVector("p",n_dim)
#     bsp_qc = QuantumCircuit(n_dim)
#     bsp_qc.h(list(range(n_dim)))

#     for i in range(5):
#         i = 0

#         for q in range(n_dim):
#             bsp_qc.rz(param.params[q],[q])
#             bsp_qc.ry(param.params[q],[q])
#         for q in range(n_dim-1):
#             bsp_qc.cx(0+i, 1+i)
#             i+=1
#         for q in range(n_dim):
#             bsp_qc.rz(param.params[q],[q])

#     print(bsp_qc)
#     return bsp_qc

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def make_bsp(n_dim):
    param = ParameterVector("p", n_dim)
    bsp_qc = QuantumCircuit(n_dim)

    bsp_qc.h(list(range(n_dim)))

    for layer in range(3):

        for q in range(n_dim):
            bsp_qc.rz(2 * param[q], q)
            bsp_qc.ry(2 * param[q], q)

        for q in range(n_dim - 1):
            bsp_qc.cx(q, q + 1)

        for q in range(n_dim):
            bsp_qc.rz(2 * param[q], q)

    print(bsp_qc)
    return bsp_qc

# seeds = np.array([44,626,5451])*1#099
# seeds = np.array([44,626])*1#099
# seeds = np.array([44,626,5451,123,4332,4255,23,234,5,4])*1#
# seeds = np.array([626,5451,123,4332,234,4,4*12])*1#
# seeds = np.array([626,5451,123,4332,234,4,4*12,1233,343,2332])*1#  #
# seeds = np.array([4*12,231*5,42**4])
seeds = np.array([231*5,626,5451,123,4332,234,4,4*12,1233,343])*1#  #,

metrics_all = []
acc_all,acc_all2 = [],[]
for noise in seeds:

    save, score, train_acc, save_score_val_2, = [], [], [], []

    data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.0028*23.5/10, random_state=noise, stratify = labels)
    rus = RandomUnderSampler(random_state=42)

    data_train, train_labels = rus.fit_resample(data_train, train_labels)#
    score = []
    system = 'val'
    
    if system == 'val':
        split_iterator = [(None, None)]
    else:
        split_iterator = kf.split(data_train, train_labels)


    for train_index, test_index in split_iterator:

        if system == 'val':
            X_train = data_train
            y_train = train_labels

            X_test = data_val
            y_test = val_labels

        else:
            X_train = data_train.iloc[train_index]
            X_test = data_train.iloc[test_index]

            y_train = train_labels.iloc[train_index]
            y_test = train_labels.iloc[test_index]

        print("Train:", Counter(y_train))
        print("Eval:", Counter(y_test))

        matrix_train_normalised = scaler.fit_transform(X_train)
        matrix_test_normalised = scaler.transform(X_test)

        num_qubits = X_train.shape[1]

        zz_kernelf = ZZFeatureMap(
            num_qubits,
            reps=1,
            entanglement='linear'
        )

        # zz_kernelf = make_bsp(num_qubits)
        zz_kernel = QuantumKernel(
            feature_map=zz_kernelf,
            quantum_instance=Aer.get_backend('statevector_simulator')
        )

        matrix_train = zz_kernel.evaluate(matrix_train_normalised)

        svc = SVC(kernel='precomputed')
        fitted = svc.fit(matrix_train, y_train)

        pred_train = fitted.predict(matrix_train)
        scor_train = balanced_accuracy_score(y_train, pred_train)
        train_acc.append(scor_train)

        matrix_test = zz_kernel.evaluate(
            x_vec=matrix_test_normalised,
            y_vec=matrix_train_normalised
        )

        pred = fitted.predict(matrix_test)

        scor = balanced_accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)

        metrics = compute_metrics(y_test, pred)
        metrics["seed"] = noise

        print("System:", system)
        print("Seed:", noise)
        print("Evaluation score:", scor)
        print(cm)

        score.append(scor)

    acc_all.append(np.mean(score))
    metrics_all.append(metrics)
        # print('Train acc: ', np.mean(scor_train))


print('Train acc: ', np.mean(scor_train))

savedf = pd.DataFrame(acc_all)


print('Acc mean val: ',np.mean(acc_all))
# print('Acc mean val: ', np.mean(acc_all2))


print(acc_all2)


metrics_seed_df = pd.DataFrame(metrics_all)

print(metrics_seed_df)

mean_metrics = metrics_seed_df.drop(columns=["seed"]).mean(numeric_only=True)
std_metrics = metrics_seed_df.drop(columns=["seed"]).std(numeric_only=True)

print(mean_metrics)
print(std_metrics)


# print(savedf)

#savedf.to_csv('save_results_qkernel.csv')
