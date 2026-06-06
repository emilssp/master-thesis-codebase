# Self-Consistency Field Calculations

This code was developed by me, working towards a master's thesis in theoretical condensed matter physics. A Python toolkit for solving the Bogoliubov-de Gennes self-consistently on two-dimensional lattice systems.

---

## Repository Structure

```
.
├── scripts/
│   ├── self_consistency_ft/     # Self-consistency solver: Fourier-transformed along both axes
│   ├── Hamiltonian.py           # Hamiltonian construction and diagonalization
│   ├── Lattice.py               # Lattice geometry definitions
│   ├── constants.py             # Physical and numerical constants
│   ├── partial_sc.py            # Same as self_consistency.py solving only part of the system keeping the rest fixed
│   ├── self_consistency.py      # Self-consistency solver for a system with PBC along y, and interfaces across x
│   └── utils.py                 # Shared utility functions
├── examples/                    # Example scripts and usage demos
├── .gitignore
└── README.md
```

---

## Getting Started

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Basic Usage

See the `examples/` directory for complete usage scenarios.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Emil Spasov