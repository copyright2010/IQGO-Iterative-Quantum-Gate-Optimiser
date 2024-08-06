import pandas as pd
import numpy as np
from sklearn.svm import SVC
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from IQGO_module.iqgo_optimize_module import IQGO, IQGO_VQC
from sklearn.model_selection import StratifiedKFold
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer

class IQGO_train():
    def __init__(self, model=None, noise_level=0.2, seed_val=42, kfold_splits = 5):
        self.noise = noise_level
        self.kf = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=128*1)
        self.scaler = MinMaxScaler()
        self.rus = RandomUnderSampler(random_state=42)
        self.seed_val = seed_val
        self.model = model

    def add_gaussian_noise(self, X, mean=0, std=0.2, seed=42):
        np.random.seed(seed)  # Set the seed
        X_noisy = X.copy()

        for column in X_noisy.columns:
            rolling_std = X_noisy[column].std()
            gaussian_noise = np.random.normal(mean, rolling_std*std, X[column].shape)
            X_noisy[column] += gaussian_noise

        return X_noisy

    def noise_ratio(self, X_noise, X_noise_free):
        noisy_signal = np.array([X_noise])
        original_signal = np.array([X_noise_free])

        # Calculate the standard deviation of the signal and the noise
        signal_std = np.std(X_noise_free)
        noise_std = np.std(noisy_signal - original_signal)

        # Calculate the noise-to-signal ratio
        nsr = noise_std / signal_std
        print(f"Noise-to-Signal Ratio: {nsr}")


    def fit(self, data_train=None, labels=None, data_val=None, val_labels=None, number_of_layers=3):

        save_all, save_all_train, save_all_test, save, compiled_circuit = [], [], [], [], []

        if data_val is None and val_labels is None:
            data_train, data_val, train_labels, val_labels = train_test_split(data_train, labels, train_size=0.66, random_state=123, stratify = labels)
        
        data_train, train_labels = self.rus.fit_resample(data_train, train_labels)

        for layer in range(number_of_layers):
            best_layer_combination = []
            print('Compiling layer', layer)

            for train_index, test_index in self.kf.split(data_train, train_labels):

                X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
                y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

                print('Train no.samples: ',Counter(y_train),'Test no.samples: ',Counter(y_test),'Val no.samples: ',Counter(val_labels))

                X_test = self.add_gaussian_noise(X_test, mean=0, std=self.noise, seed=self.seed_val)

                matrix_train_normalised = self.scaler.fit_transform(X_train)
                matrix_test_normalised = self.scaler.transform(X_test)
                
                if self.model == None:
                    self.model = SVC(kernel='precomputed')
                
                iqgo = IQGO(matrix_train_normalised, y_train)
                # iqgo.add_layer(layer_combination=1)
                # iqgo.add_layer(layer_combination=1)

                if len(compiled_circuit) != 0:
                    for combination in compiled_circuit:
                        print('add layer')
                        iqgo.add_layer(layer_combination=np.array(combination))

                mode = 'train'
                if mode == 'train':
                    accuracies_train, accuracies_test = iqgo.fit_layer(self.model, matrix_test_normalised, y_test)
                    save_all_train.append(accuracies_train)
                    save_all_test.append(accuracies_test)
                    accuracies = accuracies_test

                save_all.append(accuracies)

            column_means_train = pd.DataFrame(save_all_train).mean()
            column_means_test = pd.DataFrame(save_all_test).mean()

            both_mean = pd.DataFrame([column_means_train,column_means_test]).mean()
            column_means = pd.DataFrame(save_all).mean()  

            max_values = column_means[column_means == column_means.min()]
            print('Maximum values and their indexes test:')
            for idx, value in max_values.items():
                print(f'Index: {idx}, Value: {value}')
                best_layer_combination.append(idx)
            
            max_values = both_mean[both_mean == both_mean.min()]
            print('min values and their indexes train and test:')
            for idx, value in max_values.items():
                print(f'Index: {idx}, Value: {value}')

            save.append(column_means)

            print(best_layer_combination)
            
            compiled_circuit.append(best_layer_combination[0])

        print(compiled_circuit)

        return compiled_circuit
    
    def predict(self, data_train=None, labels=None ,data_val=None, val_labels=None, compiled_circuit=None, mode = 'val'):
        
        if data_val is None and val_labels is None:
            data_train, data_val, train_labels, val_labels = train_test_split(data_train, labels, train_size=0.66, random_state=123, stratify = labels)
        
        data_train, train_labels = self.rus.fit_resample(data_train, train_labels)
        
        save_all = []

        for train_index, test_index in self.kf.split(data_train, train_labels):

            X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            print('Train no.samples: ',Counter(y_train),'Test no.samples: ',Counter(y_test),'Val no.samples: ',Counter(val_labels))

            X_test = self.add_gaussian_noise(X_test, mean=0, std=self.noise, seed=self.seed_val)

            matrix_train_normalised = self.scaler.fit_transform(X_train)
            matrix_test_normalised = self.scaler.transform(X_test)
            matrix_val_normalised = self.scaler.transform(data_val)    

            if self.model ==  None:
                self.model = SVC(kernel='precomputed')
            
            iqgo = IQGO(matrix_train_normalised, y_train)

            if len(compiled_circuit) != 0:
                for combination in compiled_circuit:
                    print('add layer: ', combination)
                    iqgo.add_layer(layer_combination=np.array(combination))

            if mode == 'test':
                predictions = iqgo.predict(self.model, matrix_test_normalised)
                accuracies = balanced_accuracy_score(y_test, predictions)
            elif mode == 'val':
                predictions = iqgo.predict(self.model, matrix_val_normalised)
                accuracies = balanced_accuracy_score(val_labels, predictions)
            
            save_all.append(accuracies)

            column_means = pd.DataFrame(save_all)
            print(column_means)
            # max_values = column_means[column_means == column_means.max()]
            # print('Maximum values and their indexes test:')
            # for idx, value in max_values.items():
            #     print(f'Index: {idx}, Value: {value}')
        
        return predictions, column_means.mean()
    
    def compile_kernel(self, data_train=None, labels=None ,data_val=None, val_labels=None, compiled_circuit=None):

        iqgo = IQGO(data_train, labels)

        if len(compiled_circuit) != 0:
            for combination in compiled_circuit:
                print('add layer: ', combination)
                quantum_circuit = iqgo.add_layer(layer_combination=np.array(combination))
        
        kernel = QuantumKernel(feature_map=quantum_circuit, quantum_instance=Aer.get_backend('statevector_simulator'))

        matrix_train = kernel.evaluate(data_train)
        matrix_val = kernel.evaluate(x_vec=data_val, y_vec=data_train)

        return matrix_train, matrix_val
class IQGO_trainVQC():
    def __init__(self, noise_level=0.2, seed_val=42, kfold_splits = 5):
        self.noise = noise_level
        self.kf = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=128*1)
        self.scaler = MinMaxScaler()
        self.rus = RandomUnderSampler(random_state=42)
        self.seed_val = seed_val

    def add_gaussian_noise(self, X, mean=0, std=0.2, seed=42):
        np.random.seed(seed)  # Set the seed
        X_noisy = X.copy()

        for column in X_noisy.columns:
            rolling_std = X_noisy[column].std()
            gaussian_noise = np.random.normal(mean, rolling_std*std, X[column].shape)
            X_noisy[column] += gaussian_noise

        return X_noisy
    

    def noise_ratio(self, X_noise, X_noise_free):
        noisy_signal = np.array([X_noise])
        original_signal = np.array([X_noise_free])

        # Calculate the standard deviation of the signal and the noise
        signal_std = np.std(X_noise_free)
        noise_std = np.std(noisy_signal - original_signal)

        # Calculate the noise-to-signal ratio
        nsr = noise_std / signal_std
        print(f"Noise-to-Signal Ratio: {nsr}")


    def fit(self, data_train, labels, number_of_layers):

        save_all, save_all_train, save_all_test, save, compiled_circuit = [], [], [], [], []

        data_train, data_val, train_labels, val_labels = train_test_split(data_train, labels, train_size=0.66, random_state=123, stratify = labels)
        
        data_train, train_labels = self.rus.fit_resample(data_train, train_labels)

        for layer in range(number_of_layers):
            best_layer_combination = []
            print('Compiling layer', layer)

            for train_index, test_index in self.kf.split(data_train, train_labels):

                X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
                y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

                print('Train no.samples: ',Counter(y_train),'Test no.samples: ',Counter(y_test),'Val no.samples: ',Counter(val_labels))

                X_test = self.add_gaussian_noise(X_test, mean=0, std=self.noise, seed=self.seed_val)

                matrix_train_normalised = self.scaler.fit_transform(X_train)
                matrix_test_normalised = self.scaler.transform(X_test)


                #initial_point = initializer(shape=(1,ansatz.num_parameters))

               # zz_kernelf = ZZFeatureMap(data_train.shape[1], reps=3, entanglement='circular')
                
                iqgo_qvc = IQGO_VQC(matrix_train_normalised, y_train)
                # iqgo.add_layer(layer_combination=1)
                # iqgo.add_layer(layer_combination=1)

                if len(compiled_circuit) != 0:
                    for combination in compiled_circuit:
                        print('add layer')
                        iqgo_qvc.add_layer(layer_combination=np.array(combination))

                mode = 'train'
                if mode == 'train':
                    accuracies_train, accuracies_test = iqgo_qvc.fit_layer(matrix_test_normalised, y_test)
                    save_all_train.append(accuracies_train)
                    save_all_test.append(accuracies_test)
                    accuracies = accuracies_test

                save_all.append(accuracies)

            column_means_train = pd.DataFrame(save_all_train).mean()
            column_means_test = pd.DataFrame(save_all_test).mean()

            both_mean = pd.DataFrame([column_means_train,column_means_test]).mean()
            column_means = pd.DataFrame(save_all).mean()  

            max_values = column_means[column_means == column_means.min()]
            print('Maximum values and their indexes test:')
            for idx, value in max_values.items():
                print(f'Index: {idx}, Value: {value}')
                best_layer_combination.append(idx)
            
            max_values = both_mean[both_mean == both_mean.min()]
            print('min values and their indexes train and test:')
            for idx, value in max_values.items():
                print(f'Index: {idx}, Value: {value}')

            save.append(column_means)

            print(best_layer_combination)
            
            compiled_circuit.append(best_layer_combination[0])

        print(compiled_circuit)

        return compiled_circuit
    

    def predict(self, data_train, labels, compiled_circuit, mode = 'val'):

        data_train, data_val, train_labels, val_labels = train_test_split(data_train, labels, train_size=0.66, random_state=123, stratify = labels)
        
        data_train, train_labels = self.rus.fit_resample(data_train, train_labels)
        
        save_all = []

        for train_index, test_index in self.kf.split(data_train, train_labels):

            X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            print('Train no.samples: ',Counter(y_train),'Test no.samples: ',Counter(y_test),'Val no.samples: ',Counter(val_labels))

            X_test = self.add_gaussian_noise(X_test, mean=0, std=self.noise, seed=self.seed_val)

            matrix_train_normalised = self.scaler.fit_transform(X_train)
            matrix_test_normalised = self.scaler.transform(X_test)
            matrix_val_normalised = self.scaler.transform(data_val)    

            model = SVC(kernel='precomputed')
            
            iqgo_vqc = IQGO_VQC(matrix_train_normalised, y_train)

            if len(compiled_circuit) != 0:
                for combination in compiled_circuit:
                    print('add layer: ', combination)
                    iqgo_vqc.add_layer(layer_combination=np.array(combination))

            if mode == 'test':
                predictions = iqgo_vqc.predict(model, matrix_test_normalised)
                accuracies = balanced_accuracy_score(y_test, predictions)
            elif mode == 'val':
                predictions = iqgo_vqc.predict(model, matrix_val_normalised)
                accuracies = balanced_accuracy_score(val_labels, predictions)
            
            save_all.append(accuracies)

            column_means = pd.DataFrame(save_all)
            print(column_means)
            # max_values = column_means[column_means == column_means.max()]
            # print('Maximum values and their indexes test:')
            # for idx, value in max_values.items():
            #     print(f'Index: {idx}, Value: {value}')
        
        return predictions, column_means.mean()
    
