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
from sklearn.linear_model import LogisticRegression

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


svc = SVC(kernel='poly',probability=True)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=5,
    random_seed=42+3000+800,
    logging_level='Silent')

print(Counter(labels))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=128*1)
scaler = MinMaxScaler()
log_clf = LogisticRegression(solver='liblinear', random_state=42)

score, save_all, scor_train = [], [], []

data_train, data_val, train_labels, val_labels = train_test_split(data_train2, labels, train_size=0.66, random_state=123, stratify = labels)

rus = RandomUnderSampler(random_state=42)

data_train, train_labels = rus.fit_resample(data_train, train_labels)
# fold_predictions = np.zeros((len(val_labels), 5, 2))
# Number of folds in your KFold cross-validation
num_folds = kf.get_n_splits()

# Number of samples in your validation set and number of classes
num_val_samples = len(val_labels)
num_classes = len(np.unique(train_labels))

# Initialize fold_predictions with the correct shape
fold_predictions = np.zeros((num_val_samples, num_folds, num_classes))

# No need for fold_indices unless you have missing predictions in some folds
# fold_indices = np.zeros((num_val_samples, num_folds), dtype=bool)

for fold_idx, (train_index, test_index) in enumerate(kf.split(data_train, train_labels)):
    X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

    print(Counter(y_train))
    print(Counter(y_test))
    print(Counter(val_labels))

    # Normalize training data
    matrix_train_normalised = scaler.fit_transform(X_train)
    # Normalize validation data using the same scaler
    matrix_val_normalised = scaler.transform(data_val)

    # Fit your model
    fitted = log_clf.fit(matrix_train_normalised, y_train)

    # Predict probabilities on the validation set
    pred = fitted.predict_proba(matrix_val_normalised)

    # Normalize predictions to [0,1] if necessary
    y_prob = pred
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # Optional normalization

    # Store predictions
    fold_predictions[:, fold_idx, :] = y_prob

# After the cross-validation loop

# Compute average predictions and variance over folds
avg_predictions = np.mean(fold_predictions, axis=1)  # Shape: (num_val_samples, num_classes)
variance = np.var(fold_predictions, axis=1, ddof=1)  # Shape: (num_val_samples, num_classes)

# One-hot encode true labels
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
y_true_onehot = onehot_encoder.fit_transform(val_labels.values.reshape(-1, 1))  # Shape: (num_val_samples, num_classes)

# Ensure shapes match
assert avg_predictions.shape == y_true_onehot.shape, "Shape mismatch between avg_predictions and y_true_onehot"

# Calculate bias at each instance (mean over classes)
bias = np.mean((avg_predictions - y_true_onehot) ** 2, axis=1)  # Shape: (num_val_samples,)

# Calculate mean bias and variance
mean_bias = np.mean(bias)
mean_variance = np.mean(variance)

print(f"Mean Bias: {mean_bias:.4f}, Mean Variance: {mean_variance:.4f}\n")
