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


dataset = openml.datasets.get_dataset(1068)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

data = pd.DataFrame(X)
data['y'] = pd.DataFrame(y)

data2 = data.dropna(axis=0)

X = np.array(data2)
data_train = pd.DataFrame()

for i in range(1, 22):  # From 0.1 to 2.1
    column_name = f"{i/10:.1f}"
    data_train[column_name] = data2.iloc[:, i-1]

# Create df DataFrame
df = pd.DataFrame()

for i in range(1, 22):  # From 0.1 to 2.1
    column_name = f"{i/10:.1f}"
    df[column_name] = data_train[column_name]

df['labels'] = data2['y'].values

data_train2 = df.drop(columns=['labels'])
labels = df['labels']

svc = SVC(kernel='rbf')

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=4,
    random_seed=42+3000,
    logging_level='Silent')

dt_clf = DecisionTreeClassifier(random_state=42,criterion='entropy')

print(Counter(labels))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)
scaler = MinMaxScaler()
knn_clf = KNeighborsClassifier(n_neighbors=3)

score, save_all, scor_train = [], [], []

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

    train_labels_meta_1 = y_train
    test_labels_meta_1 = y_test
    val_labels_meta_1 = val_labels

    fitted = dt_clf.fit(matrix_train_normalised, train_labels_meta_1)
    pred_train = fitted.predict(matrix_train_normalised)
    
    
    scor_t = balanced_accuracy_score(train_labels_meta_1, pred_train)
    # cm = confusion_matrix(train_labels_meta_1,pred_train)
    scor_train.append(scor_t)
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
# print(Counter(pred))
    save_score = [scor]

    score.append(save_score)
    print('Train acc: ', np.mean(scor_train))

savedf = pd.DataFrame(score)
#savedf.to_csv('tabnet_test_acc_11.csv')

dfout_lgbmc = pd.DataFrame(savedf)
print(dfout_lgbmc)
print(dfout_lgbmc.mean())