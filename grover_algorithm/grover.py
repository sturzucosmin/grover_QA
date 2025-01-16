from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, AerProvider
from qiskit.visualization import plot_histogram, plot_circuit_layout, plot_gate_map
from qiskit.compiler import transpile
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, SamplerOptions
import matplotlib.pyplot as plt
import os


def create_oracle(n_qubits, marked_state):
    """
    Creates an oracle that marks a specific state by applying a phase flip.

    Args:
        n_qubits (int): Number of qubits in the circuit
        marked_state (str): Binary string representing the marked state

    Returns:
        QuantumCircuit: Oracle circuit
    """
    oracle_qc = QuantumCircuit(n_qubits)

    # Apply X gates to qubits where marked_state has '0'
    for qubit, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            oracle_qc.x(qubit)

    # Multi-controlled Z gate
    oracle_qc.h(n_qubits - 1)
    oracle_qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle_qc.h(n_qubits - 1)

    # Uncompute X gates
    for qubit, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            oracle_qc.x(qubit)

    return oracle_qc


def create_diffusion(n_qubits):
    """
    Creates the diffusion operator (amplitude amplification).

    Args:
        n_qubits (int): Number of qubits in the circuit

    Returns:
        QuantumCircuit: Diffusion circuit
    """
    qc = QuantumCircuit(n_qubits)

    # Apply H gates to all qubits
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Apply X gates to all qubits
    for qubit in range(n_qubits):
        qc.x(qubit)

    # Apply multi-controlled Z gate
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # Uncompute X gates
    for qubit in range(n_qubits):
        qc.x(qubit)

    # Uncompute H gates
    for qubit in range(n_qubits):
        qc.h(qubit)

    return qc


def grover_circuit(n_qubits, marked_state):
    """
    Creates the complete Grover's algorithm circuit.

    Args:
        n_qubits (int): Number of qubits in the circuit
        marked_state (str): Binary string representing the marked state

    Returns:
        QuantumCircuit: Complete Grover circuit
    """
    # Initialize circuit
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)

    # Initial superposition
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Calculate optimal number of iterations
    iterations = int(np.pi/4 * np.sqrt(2**n_qubits / 1))  # For single solution case

    # Apply Grover iteration
    for _ in range(iterations):
        # Oracle
        qc.compose(create_oracle(n_qubits, marked_state), inplace=True)
        # Diffusion
        qc.compose(create_diffusion(n_qubits), inplace=True)

    # Measure all qubits
    qc.measure(qr, cr)

    return qc


def plot_results(results_dict, backend_name, marked_state):
    """
    Plot the results from a backend execution.
    
    Args:
        results_dict (dict): Results from the quantum execution
        backend_name (str): Name of the backend used
        marked_state (str): The marked state we were searching for
    """
    plt.figure(figsize=(10, 6))
    plot_histogram(results_dict, title=f'Results from {backend_name}\nMarked state: {marked_state}')
    plt.tight_layout()
    plt.savefig(f'./pictures/grover_results_{backend_name}.png')
    plt.close()


def visualize_circuit(circuit, backend=None):
    """
    Visualize the quantum circuit and save it.
    
    Args:
        circuit (QuantumCircuit): The circuit to visualize
        backend (Backend, optional): Backend to show device layout
    """
    # Plot circuit
    circuit_fig = circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'}, fold=-1)
    plt.figure(circuit_fig.number)  # Make sure we're working with the circuit figure
    plt.savefig('./pictures/grover_circuit.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    if backend:
        try:
            # Plot circuit layout on device
            transpiled = transpile(circuit, backend, optimization_level=3)
            layout_fig = plot_circuit_layout(transpiled, backend)
            plt.savefig(f'./pictures/circuit_layout_{backend.name}.png', bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not plot layout for {backend.name}: {str(e)}")


def run_grover_local(n_qubits=3, marked_state="101", method="statevector"):
    """
    Runs Grover's algorithm on local simulator.
    
    Args:
        n_qubits (int): Number of qubits in the circuit
        marked_state (str): Binary string to search for
        method (str): Simulation method to use. Recommended methods are:
            - 'statevector': Most accurate but memory intensive
            - 'density_matrix': Handles mixed states, more memory intensive
            - 'matrix_product_state': Memory efficient for certain circuits
    """
    circuit = grover_circuit(n_qubits, marked_state)

    # Use specified Aer simulator
    simulator = AerSimulator(method=method)
    transpiled_circuit = transpile(circuit, simulator)

    # Run circuit
    job = simulator.run(transpiled_circuit, shots=1024)
    result = job.result()

    return result.get_counts()


def format_results(counts, marked_state):
    """
    Format the results in a readable way, highlighting the marked state.
    
    Args:
        counts (dict): Results counts from quantum execution
        marked_state (str): The marked state we were searching for
    
    Returns:
        str: Formatted string of results
    """
    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    result_str = []
    for state, count in sorted_counts:
        percentage = (count / total_shots) * 100
        if state == marked_state:
            result_str.append(f"* {state}: {count:4d} ({percentage:5.1f}%) <- marked state")
        else:
            result_str.append(f"  {state}: {count:4d} ({percentage:5.1f}%)")
    
    return "\n".join(result_str)


def check_ibm_quantum_access():
    """
    Check if IBM Quantum access is properly configured.
    Returns a tuple of (bool, str) indicating success and a message.
    """
    try:
        service = QiskitRuntimeService()
        return True, "Successfully connected to IBM Quantum"
    except Exception as e:
        if "not an IBM Quantum authentication URL" in str(e):
            return False, """
IBM Quantum authentication not configured. To use cloud backends:
1. Get an API token from https://quantum-computing.ibm.com/
2. Save it using one of these methods:
   - Run: ibmq-authenticate
   - Save token in $HOME/.qiskit/qiskitrc
   - Set QISKIT_IBM_TOKEN environment variable
"""
        return False, f"IBM Quantum error: {str(e)}"


def extract_counts_from_result(result, n_qubits, shots):
    """
    Extract counts from a quantum result object, handling different result formats.
    
    Args:
        result: The result object from job execution
        n_qubits (int): Number of qubits in the circuit
        shots (int): Number of shots used in the execution
        
    Returns:
        dict: Counts dictionary
    """
    counts = {}
    
    # Try different methods to extract the data
    try:
        # Method 1: Direct quasi_dists access
        if hasattr(result, 'quasi_dists'):
            quasi_dists = result.quasi_dists[0]
            return {format(int(k), f'0{n_qubits}b'): int(v * shots) 
                    for k, v in quasi_dists.items()}
    except Exception:
        pass

    try:
        # Method 2: Through pub_results
        pub_results = list(result)
        if pub_results:
            first_result = pub_results[0]
            raw_data = first_result.data
            
            # Try to join data if split across registers
            try:
                raw_data = first_result.join_data()
            except Exception:
                pass
            
            # Try different data formats
            if hasattr(raw_data, "to_counts"):
                return raw_data.to_counts()
            elif hasattr(raw_data, "memory"):
                for outcome in raw_data.memory:
                    counts[outcome] = counts.get(outcome, 0) + 1
                return counts
            elif hasattr(raw_data, "get_counts"):
                return raw_data.get_counts()
            elif hasattr(raw_data, "items"):
                for state, value in raw_data.items():
                    binary = format(state, f"0{n_qubits}b") if isinstance(state, int) else state
                    counts[binary] = int(value * shots) if isinstance(value, float) and 0 <= value <= 1 else value
                return counts
    except Exception as e:
        print(f"Warning: Error extracting counts: {str(e)}")
    
    return counts


if __name__ == "__main__":
    # Parameters
    N_QUBITS = 4
    MARKED_STATE = "1011"  # Example marked state for 4 qubits
    SHOTS = 1024

    os.makedirs('./pictures', exist_ok=True)

    # Create the circuit first for visualization
    circuit = grover_circuit(N_QUBITS, MARKED_STATE)
    
    # Visualize the circuit
    visualize_circuit(circuit)
    
    # Define compatible simulation methods for Grover's algorithm
    # Note: Stabilizer and extended_stabilizer methods are not included as they
    # don't support the non-Clifford gates used in Grover's algorithm
    simulation_methods = [
        "statevector",      # Most accurate simulation
        "density_matrix",   # Handles mixed states
        "matrix_product_state"  # Memory efficient
    ]
    
    print("\nRunning Grover's algorithm on local simulators:")
    print("=" * 50)
    print("Note: Using only simulators compatible with Grover's algorithm's gate set")
    
    for method in simulation_methods:
        print(f"\nSimulator Method: {method}")
        print("-" * 30)
        try:
            results = run_grover_local(N_QUBITS, MARKED_STATE, method)
            plot_results(results, f"local_{method}", MARKED_STATE)
            print("Results:")
            print(format_results(results, MARKED_STATE))
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Check IBM Quantum access before attempting cloud execution
    print("\nChecking IBM Quantum access:")
    print("=" * 50)
    has_access, message = check_ibm_quantum_access()
    
    if not has_access:
        print(message)
    else:
        try:
            # Initialize the runtime service
            service = QiskitRuntimeService()
            
            # Get all available cloud backends
            backends = [b for b in service.backends() if b.status().operational]
            
            if not backends:
                print("No operational cloud backends found.")
            else:
                print("\nRunning Grover's algorithm on cloud backends:")
                print("=" * 50)
                all_results = {}
                
                for backend in backends:
                    print(f"\nBackend: {backend.name}")
                    print("-" * 30)
                    try:
                        # Transpile circuit for this backend
                        transpiled_circuit = transpile(
                            circuit, backend=backend, optimization_level=3, seed_transpiler=42
                        )
                        
                        # Visualize circuit layout on this device
                        visualize_circuit(circuit, backend)
                        
                        # Run using sampler
                        sampler = SamplerV2(mode=backend, options=SamplerOptions(default_shots=SHOTS))
                        job = sampler.run(pubs=[transpiled_circuit], shots=SHOTS)
                        print(f"Job ID: {job.job_id()}")
                        
                        # Get results
                        result = job.result()
                        counts = extract_counts_from_result(result, N_QUBITS, SHOTS)
                        
                        if not counts:
                            print("Warning: Unable to extract counts from result")
                            print(f"Result type: {type(result)}")
                            if hasattr(result, "__dict__"):
                                print(f"Result contents: {result.__dict__}")
                            continue
                        
                        all_results[backend.name] = counts
                        
                        # Plot results for this backend
                        plot_results(counts, backend.name, MARKED_STATE)
                        print("\nResults:")
                        print(format_results(counts, MARKED_STATE))
                        
                    except Exception as e:
                        print(f"Error with backend {backend.name}: {str(e)}")
        
        except Exception as e:
            print(f"\nError running on cloud backends: {str(e)}")