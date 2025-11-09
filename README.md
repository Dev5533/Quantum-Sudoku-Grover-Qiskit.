# 🧩 Quantum Sudoku Solver using Grover’s Algorithm (Qiskit)

This project demonstrates a **4×4 Sudoku solver** implemented using **Grover’s Quantum Search Algorithm** in **IBM’s Qiskit framework**.  
It showcases how **quantum computing** can be combined with **classical logic** to efficiently solve constraint satisfaction problems through **amplitude amplification** and **quantum parallelism**.

---

## ⚙️ Project Overview

This hybrid quantum–classical algorithm performs the following:
1. **Classical Phase** – Uses backtracking to verify Sudoku constraints and find the valid solution for reference.
2. **Quantum Phase** – Encodes the unknown Sudoku cells into qubits, constructs a Grover **oracle** and **diffusion operator**, and iteratively amplifies the probability of the correct Sudoku configuration.
3. **Measurement Phase** – After running Grover iterations, the quantum system collapses into the correct Sudoku solution with high probability, verified through visual and numerical results.

---

## 🧠 Algorithmic Concept

Grover’s algorithm provides a **quadratic speedup** for unstructured search problems.  
In this Sudoku solver:

- Each unknown cell is encoded using **2 qubits** (4 possible states → numbers 1–4).  
- The **oracle** marks the correct configuration by phase inversion.  
- The **diffusion** operator performs inversion about the mean, amplifying the amplitude of the target state.  
- After ~√N iterations, measurement yields the Sudoku solution with high probability.

Mathematically, after *k* iterations:
\[
|\psi_k\rangle = \sin((2k + 1)\theta)|t\rangle + \cos((2k + 1)\theta)|t_\perp\rangle
\]
where \( \theta = \arcsin(1/\sqrt{N}) \) and \( |t\rangle \) represents the correct Sudoku state.

---

## 🧩 Example Puzzle

The initial 4×4 Sudoku puzzle used:

