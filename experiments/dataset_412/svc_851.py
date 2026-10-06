import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
from collections import Counter
import numpy as np
import openml
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import time
from scipy.stats import levy_stable

def add_gaussian_noise(X, mean=0, std=0.2, seed=42):
    np.random.seed(seed)  # Set the seed
    X_noisy = X.copy()

    for column in X_noisy.columns:
        rolling_std = X_noisy[column].std()
        gaussian_noise = np.random.normal(mean, rolling_std*std, X[column].shape)
        X_noisy[column] += gaussian_noise

    return X_noisy

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

def report(y_true, y_pred):

    cls_precision = precision_score(y_true, y_pred, average='weighted')
    cls_recall = recall_score(y_true, y_pred, average='weighted')
    cls_f1 = f1_score(y_true, y_pred, average='weighted')
    # roc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    roc = f1_score(y_true, y_pred, average='weighted')

    return cls_precision, cls_recall, cls_f1, roc


import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from collections import Counter
import numpy as np
import openml
# from IQGO_module.iqgo_training import IQGO_train
import time


data_frame = pd.read_csv('412_fig6_r2_param_estimates.csv', sep=",", index_col=None).dropna(axis=1)

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
df['0.2'] = data_train['0.2']
df['0.3'] = data_train['0.3']
df['labels'] = y

print(Counter(df['labels']),'................all labels..................')


data_train2 = df.drop(columns=['labels'])
labels2 = df['labels']


data_train2 = data_train2[25:]#.dropna()
labels = labels2[25:]

print(data_train2)
print(labels)

svc = SVC(kernel='linear')

# cat_model = CatBoostClassifier(
#     iterations=500,
#     learning_rate=0.01,
#     depth=4,
#     random_seed=42+3000+800,
#     logging_level='Silent')

cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.01,
    leaf_estimation_iterations=3,
    depth=6,
    random_seed=42+3000+800,
    bagging_temperature=0.9,
    logging_level='Silent')

print(Counter(labels))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)
scaler = MinMaxScaler()


# seeds = np.array([42**4,231,42*4,123,3223,341,222])
# seeds = np.array([12*10])
# seeds = np.array([4*12,231,42**4])#,345,353,6,34553,3646,5634,2342,234,546,564])
# seeds = np.array([1498,1498*2,1498*3,1498*4,1498*5]) #
# seeds = np.array([851,851*2,851*3,851*4,851*5]) #
# seeds = np.array([42**4,231,42*4,123,3223])
# seeds = np.array([4*12,231*5,42**4])
seeds = np.array([44,626,5451])

# from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
# import psutil
# import os

# process = psutil.Process(os.getpid())

knn_clf = KNeighborsClassifier(n_neighbors=6,leaf_size=50,weights= 'distance')
dt_clf = DecisionTreeClassifier(random_state=42,criterion='gini')
log_clf = LogisticRegression(solver='newton-cholesky', random_state=42)
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
# clfr = clf
# seeds = np.array([949,949*2,949*3,949*4,949*5]) #
# seeds = np.array([data_id])
acc_all, acc_all_test = [], []
svc_rbf = SVC(kernel='rbf', C=1, degree=3)

svc_linear = SVC(kernel='linear', C=1)

svc_poly = SVC(kernel='poly', C=1, degree=3)
# seeds = np.array([44,626])*1#099 ,5451,123,4332,4255
seeds = np.array([231*5,626,5451,123,4332,234,4,4*12,1233,343])*1#  #

for noise in seeds:
    # clf = TabPFNClassifier(random_state=noise)
    # clf = AutoTabPFNClassifier(max_time=120, device="cuda") 

    clfr = cat_model

    data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.0028*23.5/10, random_state=noise, stratify = labels)
    score, score_test, save_all, scor_train = [], [], [],[]

    rus = RandomUnderSampler(random_state=42)

    data_train, train_labels = rus.fit_resample(data_train, train_labels)
    program = 'val'

    if program == 'val':

        matrix_train_normalised = scaler.fit_transform(data_train)

        # matrix_test_meta_1 = add_gaussian_noise(matrix_test_meta_1,mean=0, std=0, seed=42)
        ns = 0.5
        gma = 0.021
        # X_train = add_levy_noise(X_train, mean=0, alpha=ns, gamma=gma, seed=noise)
        # X_test = add_levy_noise(X_test, mean=0, alpha=ns, gamma=gma, seed=noise)
        # data_val = add_levy_noise(data_val, mean=0, alpha=ns, gamma=gma, seed=noise)

        matrix_val_normalised = scaler.transform(data_val)    

        train_labels_meta_1 = train_labels
        # test_labels_meta_1 = y_test
        val_labels_meta_1 = val_labels

        fitted = clfr.fit(matrix_train_normalised, train_labels_meta_1)

        program = 'val'

        start_time = time.time()

        pred_train = fitted.predict(matrix_train_normalised)
        
        
        scor_t = balanced_accuracy_score(train_labels_meta_1, pred_train)
        # cm = confusion_matrix(train_labels_meta_1,pred_train)
        scor_train.append(scor_t)

        start_pred = time.perf_counter()        

        pred = fitted.predict(matrix_val_normalised)
        
        end = time.perf_counter()

        time_pred = end  - start_pred

        print('pred validate time: ',time_pred)

    
        cm = confusion_matrix(val_labels_meta_1, pred)

        scor_val = balanced_accuracy_score(val_labels_meta_1, pred)
        cm = confusion_matrix(val_labels_meta_1, pred)
        cls_report = classification_report(val_labels_meta_1, pred)
        
        precision, recall, f1, roc = report(val_labels, pred)

        print(cm)
        print(scor_train)
    # print(Counter(pred))

        save_score_test = [scor_val, precision, recall, f1, roc]
        score_test.append(save_score_test)


        save_score_val = [scor_val, precision, recall, f1, roc]
        score.append(save_score_val)
        # print(f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB")

    else:
        for train_index, test_index in kf.split(data_train, train_labels):
            
            X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            print(Counter(y_train))
            print(Counter(y_test))
            print(Counter(val_labels))

            matrix_train_normalised = X_train # scaler.fit_transform(X_train)

            # matrix_test_meta_1 = add_gaussian_noise(matrix_test_meta_1,mean=0, std=0, seed=42)
            ns = 0.5
            gma = 0.021
            # X_train = add_levy_noise(X_train, mean=0, alpha=ns, gamma=gma, seed=noise)
            # X_test = add_levy_noise(X_test, mean=0, alpha=ns, gamma=gma, seed=noise)
            # data_val = add_levy_noise(data_val,mean=0, alpha=ns, gamma=gma, seed=noise)

            matrix_test_normalised = X_test # scaler.transform(X_test)
            #matrix_test_normalised = scaler2.transform(matrix_test_normalised)


            matrix_val_normalised = data_val # scaler.transform(data_val)    

            train_labels_meta_1 = y_train
            test_labels_meta_1 = y_test
            val_labels_meta_1 = val_labels

            fitted = clfr.fit(matrix_train_normalised, train_labels_meta_1)

            program = 'test'

            start_time = time.time()

            pred_train = fitted.predict(matrix_train_normalised)
            
            
            scor_t = balanced_accuracy_score(train_labels_meta_1, pred_train)
            # cm = confusion_matrix(train_labels_meta_1,pred_train)
            scor_train.append(scor_t)

            if program == 'test':

                end_time_train = time.time()
                execution_time = end_time_train - start_time
                print('Training Time: ',execution_time)

                pred = fitted.predict(matrix_test_normalised)
                
                end_time_predict = time.time()
                prediction_time = end_time_predict - end_time_train
                print('execution Time: ', prediction_time)
            
                scor_test = balanced_accuracy_score(test_labels_meta_1, pred)
                cm = confusion_matrix(test_labels_meta_1, pred)
                precision_t, recall_t, f1_t, roc = report(test_labels_meta_1, pred)

            # elif program == 'val':

                end_time_train = time.time()
                execution_time = end_time_train - start_time
                print('Training Time: ',execution_time)

                pred = fitted.predict(matrix_val_normalised)
                
                end_time_predict = time.time()
                prediction_time = end_time_predict - end_time_train
                print('execution Time: ', prediction_time)
            
                cm = confusion_matrix(val_labels_meta_1, pred)

                scor_val = balanced_accuracy_score(val_labels_meta_1, pred)
                cm = confusion_matrix(val_labels_meta_1, pred)
                cls_report = classification_report(val_labels_meta_1, pred)
                
                precision, recall, f1, roc = report(val_labels, pred)

            print(cm)
            print(scor_train)
        # print(Counter(pred))
            save_score_test = [scor_test, precision_t, recall_t, f1_t, roc]

            save_score_val = [scor_val, precision, recall, f1, roc]

            score.append(save_score_val)
            score_test.append(save_score_test)

    acc_all.append(score)
    acc_all_test.append(score_test)
    # print('Train acc: ', np.mean(scor_train))

report_df = pd.DataFrame(np.vstack(acc_all), columns=['acc_val', 'Precision', 'Recall', 'F1', 'auc'])
report_df_test = pd.DataFrame(np.vstack(acc_all_test), columns=['acc_test', 'Precision', 'Recall', 'F1','auc'])


print(report_df_test)
print('Train acc: ', np.mean(scor_train))

print('Acc mean test:', report_df_test.mean())
print('Acc mean val: ', report_df.mean())

print('Acc max val: ', report_df['acc_val'].max())
print('Acc std val: ', report_df['acc_val'].std())
