
import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import HamiltonianGate
# Quantum Deep Neural Network Layer
class QDNNL():
    def __init__(self, num_qubits, D_epsilon, D_gamma, hamiltonians=[np.eye(4)]):
        self.num_qubits = num_qubits
        self.D_epsilon = D_epsilon
        self.D_gamma = D_gamma
        self.hamiltonians = [HamiltonianGate(h, 1) for h in hamiltonians]
        # inputs from most recent forward propogation
        self.inputs = None
        # parameters to encoder, stored as one dimensional vector
        # size will be: num_qubits *(2 + self.D_epsilon * 3)
        self.encoder_parameters = list()
        # parameters of transformer circuit, stored as one dimensional vector
        # size will be: num_qubits *(self.D_gamma * 3 + 2)
        self.transformer_parameters = list()
        # initialize quantum objects
        self.circuit : QuantumCircuit= QuantumCircuit(QuantumRegister(num_qubits), ClassicalRegister(num_qubits))
        self.backend = Aer.get_backend('statevector_simulator')
        # append encoder to circuit
        self.circuit = self.circuit.compose(self.build_encoder(),range(self.num_qubits))
        # append transformer to circuit
        self.circuit = self.circuit.compose(self.build_transformer(),range(self.num_qubits))
        # create all output circuits
        self.complete_circuits = list()
        for hamiltonian in self.hamiltonians:
            self.complete_circuits.append(self.circuit.compose(self.build_measurement(hamiltonian), range(self.num_qubits)))
        
        
    def build_ent(self):
        ent = QuantumCircuit(self.num_qubits)
        for qubit in range(1, self.num_qubits):
            ent.cx(qubit-1, qubit)
        ent.cx(self.num_qubits - 1, 0)
        return ent.to_instruction(label="Ent")
        
        
        """
        create encoder circuit
        :param De: an integer that indicates how many times a part of this circuit is repeated
        :return: encoder quantum circuit
        :rtype: Instruction
        """
    def build_encoder(self):
        encoder = QuantumCircuit(self.num_qubits)
        
        encoder.x(range(self.num_qubits))
        encoder.z(range(self.num_qubits))
        encoder.barrier(range(self.num_qubits))
        for _ in range(self.D_epsilon):
            encoder = encoder.compose(self.build_ent(), range(self.num_qubits))
            encoder.z(range(self.num_qubits))
            encoder.x(range(self.num_qubits))
            encoder.z(range(self.num_qubits))
            encoder.barrier(range(self.num_qubits))
            
        return encoder.to_instruction(label="encoder|D_E={}".format(self.D_epsilon))
    
    def build_transformer(self):
        transformer = QuantumCircuit(self.num_qubits)
        
        for _ in range(self.D_gamma):
            transformer = transformer.compose(self.build_ent(), range(self.num_qubits))
            transformer.z(range(self.num_qubits))
            transformer.x(range(self.num_qubits))
            transformer.z(range(self.num_qubits))
            transformer.barrier(range(self.num_qubits))
        
        transformer.x(range(self.num_qubits))
        transformer.z(range(self.num_qubits))
        
        return transformer.to_instruction(label="transformer|D_Gamma={}".format(self.D_gamma))
    
    def build_measurement(self, hamiltonian):
        measurement = QuantumCircuit(self.num_qubits)
        measurement = measurement.compose(hamiltonian)
        measurement.measure_all()
        return measurement.to_instruction(label="measurement")
    
    def forward(self, input):
        # check that all input values are zeros or ones
        for i0, i1 in zip((input > 1), input < 0):
            assert i0 == False and i1 == False
        # initialize the state of the qubits in the encoder
        self.circuit.initialize(input)
        # execute job on simulator for each complete circuit
        results = list()
        for complete_circuit in self.complete_circuits:
            results.append(self.backend.run(complete_circuit, shots=100).result())
        for result in results:
            print(result)
    
    def calculate_circuit_loss(self, vector):
        deltas = np.zeros(len(vector))
        # iterate through parameters in encoder
        for i in range(len(vector)):
            vector[i] += np.pi / 2
            h_plus = self.forward(self.inputs)
            vector[i] -= np.pi
            h_minus = self.forward(self.inputs)
            deltas[i] = (h_plus - h_minus) / 2.0
        return deltas
            
    def backpropogate_error(self, loss):
        # calculate output loss w.r.t. input
        encoder_deltas = self.calculate_circuit_loss(self.encoder_parameters)
        # iterate thorugh inputs, assign gradient for each
        transformer_deltas = self.calculate_circuit_loss(self.transformer_parameters)
        # loss with respect to inputs
        loss_inputs = loss * encoder_deltas
        # loss with respect to parameters
        loss_parameters = loss * transformer_deltas
        return loss_inputs, loss_parameters
    
class InputLayer(QDNNL):
    def __init__(self, num_qubits, D_epsilon, D_gamma):
        super().__init__(num_qubits, D_epsilon, D_gamma)
        # prepare quantum circuit
        self.input_layer = QuantumCircuit(QuantumRegister(num_qubits), ClassicalRegister(num_qubits))
        # add encoder circuit to layer
        self.input_layer = self.input_layer.compose(self.build_encoder(),range(self.num_qubits))
        self.input_layer = self.input_layer.compose(self.build_transformer(),range(self.num_qubits))
        


class QDNN():
     
    def __init__(self) -> None:
        # fixed theta used in encoding circuits
        self.theta = np.pi / 4
        pass
    
    def calculate_loss(self, output, target, type="MSE"):
        if type == "MSE":
            return np.sum(((output - target) ** 2))/len(output)