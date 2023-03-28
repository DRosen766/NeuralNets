import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import Aer, transpile

from QDNN import *



# build hamiltonians

# construct layer
input_layer = QDNNL(4, 2, 1)
input = np.zeros(16)
input[0] = 1
input_layer.forward(input)
