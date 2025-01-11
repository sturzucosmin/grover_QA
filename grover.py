from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2, SamplerOptions


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
    iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))

    # Apply Grover iteration
    for _ in range(iterations):
        # Oracle
        qc.compose(create_oracle(n_qubits, marked_state), inplace=True)
        # Diffusion
        qc.compose(create_diffusion(n_qubits), inplace=True)

    # Measure all qubits
    qc.measure(qr, cr)

    return qc


def list_simulators():
    """Lists all available local and cloud simulators"""
    try:
        service = QiskitRuntimeService()
        print("\nAvailable IBM backends:")
        for backend in service.backends():
            print(f"- {backend.name}: {backend.status}")
    except Exception as e:
        print("\nNote: IBM Runtime service not configured. To access cloud backends:")
        print("1. Register at quantum-computing.ibm.com")
        print("2. Get API token")
        print(
            "3. Run: QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"
        )


def run_grover_cloud(n_qubits=3, marked_state="101", backend_name=None):
    """
    Runs Grover's algorithm using IBM Runtime V2 Sampler.

    Args:
        n_qubits (int): Number of qubits to use
        marked_state (str): Binary string to search for
        backend_name (str): Name of the backend to use (optional)
    """
    # Create circuit
    circuit = grover_circuit(n_qubits, marked_state)

    try:
        # Initialize the runtime service
        service = QiskitRuntimeService()

        # Get backend
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.backend("ibmq_qasm_simulator")

        print(f"\nUsing backend: {backend.name}")

        # Transpile circuit for the target backend
        transpiled_circuit = transpile(
            circuit, backend=backend, optimization_level=3, seed_transpiler=42
        )

        with Session(backend=backend) as session:
            sampler = SamplerV2(
                mode=session, options=SamplerOptions(default_shots=1024)
            )

            print("Submitting job...")
            job = sampler.run(pubs=[transpiled_circuit], shots=1024)
            print(f"Job ID: {job.job_id()}")

            # Get results
            result = job.result()
            pub_results = list(result)
            if not pub_results:
                print("No pub results found")
                return None

            first_result = pub_results[0]
            print(f"First result data type: {type(first_result.data)}")

            # Get the raw data from the result
            raw_data = first_result.data

            # Try to join data if it's split across registers
            try:
                joined_data = first_result.join_data()
                print(f"Joined data type: {type(joined_data)}")
                raw_data = joined_data
            except Exception as e:
                print(f"Note: Could not join data: {str(e)}")

            counts = {}

            # Convert BitArray or array data to counts
            if hasattr(raw_data, "to_counts"):
                counts = raw_data.to_counts()
            elif hasattr(raw_data, "memory"):
                # Handle bit string memory
                for outcome in raw_data.memory:
                    counts[outcome] = counts.get(outcome, 0) + 1
            elif hasattr(raw_data, "get_counts"):
                counts = raw_data.get_counts()
            else:
                # Try to interpret raw data as probabilities or counts
                print(f"Raw data attributes: {dir(raw_data)}")
                if hasattr(raw_data, "items"):
                    for state, value in raw_data.items():
                        if isinstance(state, int):
                            binary = format(state, f"0{n_qubits}b")
                        else:
                            binary = state
                        # Check if value is probability or count
                        if isinstance(value, float) and 0 <= value <= 1:
                            counts[binary] = int(
                                value * 1024
                            )  # Convert probability to count
                        else:
                            counts[binary] = value

            if not counts:
                print(f"Warning: Unable to extract counts from data")
                print(f"Raw data type: {type(raw_data)}")
                if hasattr(raw_data, "__dict__"):
                    print(f"Raw data contents: {raw_data.__dict__}")

            return counts

    except Exception as e:
        print(f"Error in quantum circuit execution: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return None

    except Exception as e:
        print("Error accessing IBM backend:", str(e))
        print("\nTrying local simulator instead...")
        return run_grover_local(n_qubits, marked_state)


def run_grover_local(n_qubits=3, marked_state="101"):
    """
    Runs Grover's algorithm on local simulator.
    """
    circuit = grover_circuit(n_qubits, marked_state)

    # Use local simulator
    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)

    # Run circuit
    job = simulator.run(transpiled_circuit, shots=1024)
    result = job.result()

    return result.get_counts()


if __name__ == "__main__":
    # List available backends
    list_simulators()

    # Run circuit
    print("\nRunning Grover's algorithm...")
    try:
        # Try cloud backend first
        results = run_grover_cloud(3, "111", "ibm_kyiv")
        if results:
            print("\nResults:", results)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nFalling back to local simulator...")
        results = run_grover_local(3, "111")
        print("\nLocal results:", results)
