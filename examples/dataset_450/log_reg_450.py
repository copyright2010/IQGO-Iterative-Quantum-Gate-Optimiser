import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

dataset = openml.datasets.get_dataset(450)
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
data_train['0.4'] = (X[:,3])

df = pd.DataFrame()

df['0.1'] = np.log(data_train['0.1'])
df['0.2'] = data_train['0.2']
df['0.3'] = data_train['0.3']
df['0.4'] = data_train['0.4']

df['labels'] = X[:,4]

data_train2 = df.drop(columns=['labels'])
labels = df['labels']

svc = SVC(kernel='poly')

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=4,
    random_seed=42+3000,
    logging_level='Silent')

log_clf = LogisticRegression(solver='newton-cholesky', random_state=42)

print(Counter(labels))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)
scaler = MinMaxScaler()


seeds = np.array([42**4,231,42*4,123,3223,341,222])
# seeds = np.array([12*10])

acc_all = []
for noise in seeds:

    data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.66, random_state=noise, stratify = labels)
    score, save_all, scor_train = [], [], []

    rus = RandomUnderSampler(random_state=42)

    data_train, train_labels = rus.fit_resample(data_train, train_labels)

    for train_index, test_index in kf.split(data_train, train_labels):
        
        X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
        y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

        print(Counter(y_train))
        print(Counter(y_test))
        print(Counter(val_labels))

        matrix_train_normalised = scaler.fit_transform(X_train)

        # matrix_test_meta_1 = add_gaussian_noise(matrix_test_meta_1,mean=0, std=0, seed=42)

        # matrix_validate_meta_1 = add_gaussian_noise(matrix_validate_meta_1,mean=0, std=0.0, seed=23258)


        matrix_test_normalised = scaler.transform(X_test)
        #matrix_test_normalised = scaler2.transform(matrix_test_normalised)


        matrix_val_normalised = scaler.transform(data_val)    

        train_labels_meta_1 = y_train
        test_labels_meta_1 = y_test
        val_labels_meta_1 = val_labels

        fitted = log_clf.fit(matrix_train_normalised, train_labels_meta_1)

        program = 'val'

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
        
            scor = balanced_accuracy_score(test_labels_meta_1, pred)
            cm = confusion_matrix(test_labels_meta_1, pred)

        elif program == 'val':

            end_time_train = time.time()
            execution_time = end_time_train - start_time
            print('Training Time: ',execution_time)

            pred = fitted.predict(matrix_val_normalised)
            
            end_time_predict = time.time()
            prediction_time = end_time_predict - end_time_train
            print('execution Time: ', prediction_time)
        
            scor = balanced_accuracy_score(val_labels_meta_1, pred)
            cm = confusion_matrix(val_labels_meta_1, pred)

        print(scor)
        print(cm)
        print(scor_train)
    # print(Counter(pred))
        save_score = [scor]

        score.append(save_score)
    
    acc_all.append(np.mean(score))

    # print('Train acc: ', np.mean(scor_train))


print('Train acc: ', np.mean(scor_train))

savedf = pd.DataFrame(acc_all)
print('Acc mean val: ',np.mean(savedf))