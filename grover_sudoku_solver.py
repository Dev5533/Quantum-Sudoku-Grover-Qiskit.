!pip install qiskit qiskit-aer
%matplotlib inline
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import traceback

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram


def is_valid_4x4(board, r, c, v):
    for i in range(4):
        if board[r][i] == v or board[i][c] == v:
            return False
    sr, sc = (r // 2) * 2, (c // 2) * 2
    for i in range(2):
        for j in range(2):
            if board[sr + i][sc + j] == v:
                return False
    return True


def solve_4x4_classical(board):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                for v in range(1, 5):
                    if is_valid_4x4(board, r, c, v):
                        board[r][c] = v
                        if solve_4x4_classical(board):
                            return True
                        board[r][c] = 0
                return False
    return True


def val_to_bits_2(v):
    v -= 1
    return [(v >> 0) & 1, (v >> 1) & 1]


def bits_to_val_2(b0, b1):
    return (b0 | (b1 << 1)) + 1


def build_single_target_oracle(num_qubits, target_bits):
    qc = QuantumCircuit(num_qubits)
    for i, b in enumerate(target_bits):
        if b == 0:
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
    qc.h(num_qubits - 1)
    for i, b in enumerate(target_bits):
        if b == 0:
            qc.x(i)
    return qc.to_gate(label="Oracle")


def build_diffusion(num_qubits):
    d = QuantumCircuit(num_qubits)
    d.h(range(num_qubits))
    d.x(range(num_qubits))
    d.h(num_qubits - 1)
    d.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
    d.h(num_qubits - 1)
    d.x(range(num_qubits))
    d.h(range(num_qubits))
    return d.to_gate(label="Diffusion")


def decode_solution_from_bitstring(bitstring, unknown_positions, initial_board):
    num_qubits = 2 * len(unknown_positions)
    bits_by_qubit = [int(bitstring[num_qubits - 1 - i]) for i in range(num_qubits)]
    solved = deepcopy(initial_board)
    for k, (r, c) in enumerate(unknown_positions):
        b0 = bits_by_qubit[2 * k]
        b1 = bits_by_qubit[2 * k + 1]
        solved[r][c] = bits_to_val_2(b0, b1)
    return solved


def board_to_multiline_str(board):
    return "\n".join(" ".join(str(x) for x in row) for row in board)


def plot_top_solutions_pretty(counts, unknown_positions, initial_board, top_n=5, figsize=(12, 6), savepath=None):
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    labels, freqs = [], []
    for bitstring, freq in top_items:
        board = decode_solution_from_bitstring(bitstring, unknown_positions, initial_board)
        labels.append(board_to_multiline_str(board))
        freqs.append(freq)
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(freqs)), freqs, color='skyblue', edgecolor='black')
    plt.ylabel("Counts")
    plt.title(f"Top {top_n} Grover Sudoku Solutions (decoded boards)")
    plt.xticks(range(len(labels)), labels, fontsize=9)
    for rect, freq in zip(bars, freqs):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, h + max(freqs) * 0.01, str(freq), ha='center', va='bottom')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print("Saved pretty plot to", savepath)
    plt.show()


def stepwise_grover_demo(initial_board, max_unknown_cells=8, shots=4096, show_iterations=True):
    try:
        print("Initial Sudoku puzzle (0 denotes unknown):")
        for row in initial_board:
            print(row)

        board = deepcopy(initial_board)
        ok = solve_4x4_classical(board)
        if not ok:
            raise ValueError("No classical solution found or puzzle ambiguous.")
        print("\nClassical solver result (ground truth):")
        for row in board:
            print(row)

        unknown_positions = [(r, c) for r in range(4) for c in range(4) if initial_board[r][c] == 0]
        U = len(unknown_positions)
        print("\nUnknown positions (row, col):", unknown_positions)
        if U == 0:
            print("No unknowns — puzzle already complete.")
            return

        num_qubits = 2 * U
        print("Number of qubits (2 per unknown):", num_qubits)

        target_bits = [0] * num_qubits
        for k, (r, c) in enumerate(unknown_positions):
            v = board[r][c]
            b0, b1 = val_to_bits_2(v)
            target_bits[2 * k] = b0
            target_bits[2 * k + 1] = b1
        target_bitstring_qubitorder = "".join(str(b) for b in reversed(target_bits))
        print("Target bits (LSB-first per cell):", target_bits)
        print("Target bitstring (Qiskit measured order, MSB left):", target_bitstring_qubitorder)

        oracle = build_single_target_oracle(num_qubits, target_bits)
        diffusion = build_diffusion(num_qubits)

        qc_init = QuantumCircuit(num_qubits)
        qc_init.h(range(num_qubits))
        sv_init = Statevector.from_instruction(qc_init)
        probs_init = sv_init.probabilities_dict()
        p_target_init = probs_init.get(target_bitstring_qubitorder, 0.0)
        print(f"\nAfter initialization: target probability = {p_target_init:.6f}")

        top_init = sorted(probs_init.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top-5 basis states (prob, state) after init:")
        for s, p in top_init:
            print(f"  {s} : {p:.6f}")

        qc_or = QuantumCircuit(num_qubits)
        qc_or.h(range(num_qubits))
        qc_or.append(oracle, range(num_qubits))
        sv_or = Statevector.from_instruction(qc_or)
        probs_or = sv_or.probabilities_dict()
        p_target_or = probs_or.get(target_bitstring_qubitorder, 0.0)
        print(f"\nAfter applying Oracle (one call): target probability = {p_target_or:.6f}")

        top_or = sorted(probs_or.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top-5 basis states after Oracle:")
        for s, p in top_or:
            print(f"  {s} : {p:.6f}")

        qc_diff = QuantumCircuit(num_qubits)
        qc_diff.h(range(num_qubits))
        qc_diff.append(diffusion, range(num_qubits))
        sv_diff = Statevector.from_instruction(qc_diff)
        probs_diff = sv_diff.probabilities_dict()
        p_target_diff = probs_diff.get(target_bitstring_qubitorder, 0.0)
        print(f"\nAfter applying Diffusion (on init): target probability = {p_target_diff:.6f}")

        N = 4 ** U
        k_iters = max(1, int(math.floor((math.pi / 4) * math.sqrt(N))))
        print(f"\nGrover expected iterations (heuristic) k = {k_iters}")

        qc_step = QuantumCircuit(num_qubits)
        qc_step.h(range(num_qubits))
        iter_info = []

        for i in range(k_iters):
            qc_step.append(oracle, range(num_qubits))
            sv_after_or = Statevector.from_instruction(qc_step)
            probs_after_or = sv_after_or.probabilities_dict()
            p_target_after_or = probs_after_or.get(target_bitstring_qubitorder, 0.0)

            qc_step.append(diffusion, range(num_qubits))
            sv_after_do = Statevector.from_instruction(qc_step)
            probs_after_do = sv_after_do.probabilities_dict()
            p_target_after_do = probs_after_do.get(target_bitstring_qubitorder, 0.0)

            iter_info.append({
                "iter": i + 1,
                "p_after_or": p_target_after_or,
                "p_after_do": p_target_after_do,
                "top_after_do": sorted(probs_after_do.items(), key=lambda x: x[1], reverse=True)[:5]
            })

            if show_iterations:
                print(f"\nIteration {i + 1}:")
                print(f"  Target prob after Oracle: {p_target_after_or:.6f}")
                print(f"  Target prob after Diffusion: {p_target_after_do:.6f}")
                print("  Top-3 states after Diffusion:")
                for s, p in iter_info[-1]["top_after_do"][:3]:
                    print(f"    {s} : {p:.6f}")

        qc_final = qc_step.copy()
        qc_final.measure_all()
        sim = AerSimulator()
        compiled = transpile(qc_final, sim)
        job = sim.run(compiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("\nMeasurement counts (top 10):")
        for s, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {s} : {c}")

        top_bitstring = max(counts.items(), key=lambda x: x[1])[0]
        solved_board = decode_solution_from_bitstring(top_bitstring, unknown_positions, initial_board)
        print("\nSolved board decoded from top measured bitstring:")
        for row in solved_board:
            print(row)

        print("\nPlotting histogram (Figure):")
        fig = plot_histogram(counts, figsize=(10, 5))
        display(fig)

        plot_top_solutions_pretty(counts, unknown_positions, initial_board, top_n=4, savepath="top_solutions.png")

        return {
            "initial_sv_probs": probs_init,
            "after_or_probs": probs_or,
            "after_diff_probs": probs_diff,
            "iteration_info": iter_info,
            "counts": counts,
            "solved_board": solved_board,
            "target_bitstring": target_bitstring_qubitorder,
            "qc_final": qc_final
        }

    except Exception as e:
        print("Exception in stepwise_grover_demo:")
        traceback.print_exc()
        raise


puzzle = [
    [2, 0, 3, 4],
    [3, 4, 0, 1],
    [1, 0, 4, 2],
    [0, 2, 1, 3],
]

res = stepwise_grover_demo(puzzle, max_unknown_cells=8, shots=4096, show_iterations=True)
