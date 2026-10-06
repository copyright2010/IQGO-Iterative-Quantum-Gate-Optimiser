from itertools import product
from itertools import cycle

class BaseIQGO:
    def __init__(self, number_of_qubits, combination, quantum_circuit, parameter_vector):
        self.qubits = number_of_qubits
        self.combination = combination
        self.parameter_vector = parameter_vector
        self.quantum_circuit = quantum_circuit

        self.gate_funcs = [self.apply_rx, self.apply_ry, self.apply_rz]#self.apply_p
        self.combinations_all = list(product(self.gate_funcs, repeat=3))
        self.max_index = len(self.combinations_all[self.combination])
        self.pattern = [i % self.max_index for i in range(3)]
        self.pattern_cycle = cycle(self.pattern)
        self.multiplier = 2

    def apply_rx(self, circuit, qubit, theta):
        circuit.rx(self.multiplier*theta, qubit)

    def apply_ry(self, circuit, qubit, theta):
        circuit.ry(self.multiplier*theta, qubit)

    def apply_rz(self, circuit, qubit, theta):
        circuit.rz(self.multiplier*theta, qubit)

    def apply_p(self, circuit, qubit, theta):
        circuit.p(self.multiplier*theta, qubit)
    
    def feature_map(self):
        combi_selected_layer_1 = [self.combinations_all[self.combination][next(self.pattern_cycle)] for _ in range(len(self.parameter_vector))]
        for i, gate_func in enumerate(combi_selected_layer_1):
            gate_func(self.quantum_circuit, i, self.parameter_vector[i])

    def compile_circuit(self):
        self.feature_map()
        return self.quantum_circuit
