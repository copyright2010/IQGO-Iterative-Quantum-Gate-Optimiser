from IQGO_module.BaseIQGO import BaseIQGO
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import Aer
from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from qiskit.circuit.library import RealAmplitudes
from IPython.display import clear_output
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals
from sklearn.preprocessing import OneHotEncoder

criterion = nn.CrossEntropyLoss()

class IQGO:
    def __init__(self, X_train, y_train, training_noise = None):
        self.num_qubits = X_train.shape[1]
        self.parameter_vector = ParameterVector('x', self.num_qubits)
        self.quantum_circuit = QuantumCircuit(self.num_qubits)  # Initialize with a fresh circuit
        self.X_train = X_train  # Store X_train for kernel evaluation
        self.y_train = y_train
        self.training_noise = training_noise
    
    def regularized_kernel_matrix(self, kernel_matrix):
        lambda_reg = 0.05
        # Add L2 regularization term to the kernel matrix
        return kernel_matrix + lambda_reg * np.eye(kernel_matrix.shape[0], kernel_matrix.shape[1])
    
    def kernel_alignment(self, K, Y):
        lambda_reg = 0.05
        YYT = np.outer(Y, Y)
        alignment = np.sum(K * YYT) / np.sqrt(np.sum(K * K) * np.sum(YYT * YYT))
        return K + lambda_reg * (1 - alignment) * np.eye(K.shape[0], K.shape[1])
    
    def cross_entropy_loss(self, y_true, y_pred):
        # Ensure y_pred and y_true are tensors
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(np.array(y_true), dtype=torch.float32)
        
        # Use PyTorch's cross-entropy loss
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_pred, y_true)    
    
    def fit_layer(self, model=None, X_test=None, y_test=None):
        self.model = model

        if not self.quantum_circuit.data:
            for i in range(self.num_qubits):
                self.quantum_circuit.h(i)
        
        save_score_train = []
        save_score_test = []

        for comb in range(0, 27, 1):
            # Create a copy of the current quantum circuit
            quantum_circuit_copy = self.quantum_circuit.copy()
            
            # Add the combination layer
            qc = BaseIQGO(number_of_qubits=self.num_qubits, combination=comb, quantum_circuit=quantum_circuit_copy, parameter_vector=self.parameter_vector).compile_circuit()
                    
            # Generate the quantum circuit for the feature map
            print(qc.draw())
            
            # Create a quantum kernel with the feature map
            self.kernel = QuantumKernel(feature_map=qc, quantum_instance=Aer.get_backend('statevector_simulator'))
            # Evaluate the kernel matrix for training
            
            matrix_train = self.kernel.evaluate(self.X_train)
            #matrix_train = self.kernel_alignment(matrix_train, self.y_train)

            # Fit the model using the training kernel matrix
            model.fit(matrix_train, self.y_train)

            if X_test is not None and y_test is not None:
                # Evaluate the kernel matrix for testing
                #X_test = self.add_gaussian_noise(X_test, mean=0, std=self.training_noise)

                matrix_test = self.kernel.evaluate(x_vec=X_test, y_vec=self.X_train)
                #matrix_test = self.regularized_kernel_matrix(matrix_test)

                # Predict using the fitted model
                predictions_train = model.predict(matrix_train)
                predictions_test = model.predict(matrix_test)

                # Calculate and print the balanced accuracy score
                score_train = self.cross_entropy_loss(self.y_train,predictions_train)
                score_test = self.cross_entropy_loss(y_test,predictions_test)

                print(f'Combination {comb} Balanced accuracy train-test :', score_train.item(),score_test.item())

                save_score_train.append(score_train.item())
                save_score_test.append(score_test.item())

        return save_score_train, save_score_test

    def add_layer(self, layer_combination):
        if not self.quantum_circuit.data:
                for i in range(self.num_qubits):
                     self.quantum_circuit.h(i)
        try:
            if layer_combination is not None:
                self.quantum_circuit = BaseIQGO(number_of_qubits=self.num_qubits, combination=layer_combination, quantum_circuit=self.quantum_circuit, \
                                                parameter_vector=self.parameter_vector).compile_circuit()
        except:
            print('add layer_combination')

    def add_gate(self):

        if not self.quantum_circuit.data:
                for i in range(self.num_qubits):
                     self.quantum_circuit.h(i)
        try:
            # if layer_combination is not None:
            #     self.quantum_circuit = BaseIQGO(number_of_qubits=self.num_qubits, combination=layer_combination, quantum_circuit=self.quantum_circuit, parameter_vector=self.parameter_vector).get_circuit()
            for i in range(self.num_qubits):
                self.quantum_circuit.z(i)
        except:
            print('add layer_combination')

    def predict(self, model, X_val):
        # if self.kernel is None or self.X_train is None:
        #     raise ValueError("The model has not been fitted yet. Please call 'fit_layer' first.")
        #print(self.quantum_circuit.draw())
        self.kernel = QuantumKernel(feature_map=self.quantum_circuit, quantum_instance=Aer.get_backend('statevector_simulator'))

        matrix_train = self.kernel.evaluate(self.X_train)
        #matrix_train = self.kernel_alignment(matrix_train, self.y_train)
        # matrix_train = self.regularized_kernel_matrix(matrix_train)

        # Evaluate the kernel matrix for testing
        matrix_val = self.kernel.evaluate(x_vec=X_val, y_vec=self.X_train)
        #matrix_val = self.regularized_kernel_matrix(matrix_val)

        # Predict using the fitted model
        model.fit(matrix_train, self.y_train)

        # Predict using the fitted model
        predictions = model.predict(matrix_val)
        
        return predictions


class IQGO_VQC:
    def __init__(self, X_train, y_train, training_noise = None):
        self.num_qubits = X_train.shape[1]
        self.parameter_vector = ParameterVector('x', self.num_qubits)
        self.quantum_circuit = QuantumCircuit(self.num_qubits)  # Initialize with a fresh circuit
        self.X_train = X_train  # Store X_train for kernel evaluation
        self.y_train = y_train
        self.training_noise = training_noise
    
    def cross_entropy_loss(self, y_true, y_pred):
        # Ensure y_pred and y_true are tensors
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y_true = torch.tensor(np.array(y_true), dtype=torch.float32)
        
        # Use PyTorch's cross-entropy loss
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_pred, y_true)    
        #print(len(objective_func_vals))

    def fit_layer(self, X_test=None, y_test=None):
        
        def callback_graph(weights, obj_func_eval):

            clear_output(wait=False)
            objective_func_vals.append(obj_func_eval)

        if not self.quantum_circuit.data:
            for i in range(self.num_qubits):
                self.quantum_circuit.h(i)
        
        objective_func_vals = []

        save_score_train = []
        save_score_test = []

        for comb in range(0, 27, 1):
            # Create a copy of the current quantum circuit
            quantum_circuit_copy = self.quantum_circuit.copy()
            
            # Add the combination layer
            qc = BaseIQGO(number_of_qubits=self.num_qubits, combination=comb, quantum_circuit=quantum_circuit_copy, parameter_vector=self.parameter_vector).compile_circuit()
                    
            # Generate the quantum circuit for the feature map
           # print(qc.draw())

            # Create a quantum kernel with the feature map
            #self.kernel = QuantumKernel(feature_map=qc, quantum_instance=Aer.get_backend('statevector_simulator'))
            # Evaluate the kernel matrix for training
            
            #matrix_train = self.kernel.evaluate(self.X_train)
            #matrix_train = self.kernel_alignment(matrix_train, self.y_train)

            backend = Aer.get_backend('statevector_simulator')
            seed = algorithm_globals.random_seed = 1234

            quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

            ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3,entanglement='full')

            #initializer = tf.keras.initializers.random_uniform(minval=0.15, maxval=0.85, seed= 111)         
            optimizer = COBYLA(maxiter=40)

            model = VQC(
                feature_map=qc,
                ansatz=ansatz,
                optimizer=optimizer,
                quantum_instance=quantum_instance,
                callback=callback_graph,
            # initial_point=initial_point
            )
                

            encoder = OneHotEncoder()

            y_train_encoded = encoder.fit_transform(pd.DataFrame(self.y_train).values.reshape(-1, 1))
            #y_test_encoded  = encoder.transform(pd.DataFrame(y_test).values.reshape(-1, 1))

            # Fit the model using the training kernel matrix
            model.fit(self.X_train, y_train_encoded)

            if X_test is not None and y_test is not None:
                # Evaluate the kernel matrix for testing
                #X_test = self.add_gaussian_noise(X_test, mean=0, std=self.training_noise)

                #matrix_test = self.kernel.evaluate(x_vec=X_test, y_vec=self.X_train)
                #matrix_test = self.regularized_kernel_matrix(matrix_test)

                # Predict using the fitted model
                predictions_train = model.predict(self.X_train)
                predictions_test = model.predict(X_test)

                predictions_train = encoder.inverse_transform(predictions_train)
                predictions_test = encoder.inverse_transform(predictions_test)
                
                #print(predictions_train)

                # Calculate and print the balanced accuracy score
                score_train = self.cross_entropy_loss(self.y_train,predictions_train.flatten())
                score_test = self.cross_entropy_loss(y_test,predictions_test.flatten())

                print(f'Combination {comb} Balanced accuracy train-test :', score_train.item(),score_test.item())

                save_score_train.append(score_train.item())
                save_score_test.append(score_test.item())

        return save_score_train, save_score_test

    def add_layer(self, layer_combination):
        if not self.quantum_circuit.data:
                for i in range(self.num_qubits):
                     self.quantum_circuit.h(i)
        try:
            if layer_combination is not None:
                self.quantum_circuit = BaseIQGO(number_of_qubits=self.num_qubits, combination=layer_combination, quantum_circuit=self.quantum_circuit, \
                                                parameter_vector=self.parameter_vector).compile_circuit()
        except:
            print('add layer_combination')

    def add_gate(self):

        if not self.quantum_circuit.data:
                for i in range(self.num_qubits):
                     self.quantum_circuit.h(i)
        try:
            # if layer_combination is not None:
            #     self.quantum_circuit = BaseIQGO(number_of_qubits=self.num_qubits, combination=layer_combination, quantum_circuit=self.quantum_circuit, parameter_vector=self.parameter_vector).get_circuit()
            for i in range(self.num_qubits):
                self.quantum_circuit.z(i)
        except:
            print('add layer_combination')

    def predict(self, model, X_val):
        # if self.kernel is None or self.X_train is None:
        #     raise ValueError("The model has not been fitted yet. Please call 'fit_layer' first.")
        #print(self.quantum_circuit.draw())
        def callback_graph(weights, obj_func_eval):

            clear_output(wait=False)
            objective_func_vals.append(obj_func_eval)
    
        objective_func_vals= []
        
        
        backend = Aer.get_backend('statevector_simulator')
        seed = algorithm_globals.random_seed = 1234

        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3,entanglement='full')

        #initializer = tf.keras.initializers.random_uniform(minval=0.15, maxval=0.85, seed= 111)         
        optimizer = COBYLA(maxiter=40)

        model = VQC(
            feature_map=self.quantum_circuit,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            callback=callback_graph,
        # initial_point=initial_point
        )

        encoder = OneHotEncoder()

        y_train_encoded = encoder.fit_transform(pd.DataFrame(self.y_train).values.reshape(-1, 1))
        #y_test_encoded  = encoder.transform(pd.DataFrame(y_test).values.reshape(-1, 1))

        # Fit the model using the training kernel matrix
        model.fit(self.X_train, y_train_encoded)

        # Predict using the fitted model
        predictions_train = model.predict(X_val)

        predictions = encoder.inverse_transform(predictions_train)
        
        return predictions.flatten()

    # def spectral_regularization(self, K, lambda_reg=0.01):
       
        
    #     U, S, Vt  = np.linalg.svd(K, full_matrices=False)
    #     # Apply regularization to the singular values
    #     S = np.maximum(S - lambda_reg, 0)

    #     regularized_matrix = np.dot(U, np.dot(np.diag(S), Vt))        
        
    #     return regularized_matrix

    # def manifold_regularization(self, K, X, lambda_reg=0.01):
    #     """
    #     Apply manifold regularization to the kernel matrix.
        
    #     K: Kernel matrix (assumed to be a numerical matrix, e.g., a NumPy array)
    #     X: Input data matrix
    #     lambda_reg: Regularization parameter
    #     """
    #     # Ensure K is a NumPy array
    #     if not isinstance(K, np.ndarray):
    #         K = np.array(K)
        
    #     # Compute the connectivity graph
    #     W = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=True)
        
    #     # Convert the sparse matrix to a dense format
    #     W_dense = W.toarray()

    #     # Compute the degree matrix
    #     D = np.diag(np.sum(W_dense, axis=0))        
        
    #     # Compute the Laplacian matrix
    #     L = D - W_dense
        
    #     # Apply manifold regularization to the kernel matrix
    #     K_reg = K + lambda_reg * L

    #     return K_reg
