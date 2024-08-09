# IQGO Module

## Installation
To install the IQGO module, run the following command:

```sh
pip install IQGO-module
```

## Getting Started
```python
from sklearn.datasets import load_iris
from IQGO_module.iqgo_training import IQGO_train
import numpy as np
import pandas as pd

X, y = load_iris(return_X_y=True, as_frame=True)

acc_test, acc_val, save_noise, cct = [], [], [], []

# test multiple seed values for gaussian noise
seeds = np.array([3646,5634,2342,234,546,564])

for noise in seeds:

    # initialise IQGO with the noise level, noise random seed and the kfold splits
    init_iqgo = IQGO_train(noise_level = 0.2, seed_val=noise, kfold_splits=5)

    fitted_circuit = init_iqgo.fit(data_train=X, labels=y, number_of_layers = 3)

    # hardcode the best compiled circuit for testing 
    #fitted_circuit = np.array([13, 26, 8]) #best

    predictions, column_means = init_iqgo.predict(data_train = X, labels = y, compiled_circuit=fitted_circuit,mode='test')
    acc_test.append(column_means)
    
    predictions, column_means = init_iqgo.predict(data_train = X, labels = y, compiled_circuit=fitted_circuit,mode='val')
    
    acc_val.append(column_means)
    save_noise.append(noise)
    cct.append(fitted_circuit)
    print('done')
    print(fitted_circuit)


print('Acc test: ',acc_test)
print('Acc val: ',acc_val)

save_noise = np.transpose(save_noise)

print(save_noise)
all = pd.DataFrame([save_noise, acc_test, acc_val, cct])

# save the compiled circuits to select the best one
all.to_csv('compiled_circuits.csv')
```
