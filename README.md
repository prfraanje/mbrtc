# Model-Based Real-Time Control (mbrtc)

Python library and course materials for Model-Based Real-Time Control, based on the textbook "Computer-Controlled Systems" by Åström and Wittenmark.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## The mbrtc Module

The [mbrtc.py](./mbrtc.py) module provides comprehensive functions for analysis, simulation, and design of linear discrete-time and continuous-time control systems.

### Main Features

- **Signal Generation**: Functions to generate test signals (spike, step, impulse, random)
- **Continuous/Discrete Conversion**: Zero-order hold (ZOH) conversions between continuous and discrete time
- **System Simulation**: Simulate both continuous and discrete-time state-space models
- **System Analysis**: Controllability, observability, and stability testing
- **Control Design**: Pole placement for state-feedback control
- **Canonical Forms**: Transform systems to controller or observer canonical forms
- **System Interconnections**: Series, parallel, and feedback connections

### Quick Start

```python
import numpy as np
from mbrtc import *

# Create a continuous-time first-order system and discretize
Ac = np.array([[-1.0]])
Bc = np.array([[1.0]])
Cc = np.array([[1.0]])
Dc = np.array([[0.0]])
h = 0.1  # sampling time

Ad, Bd, Cd, Dd = c2d_zoh(Ac, Bc, Cc, Dc, h)

# Simulate step response
u = step_signal(NS=50)
y = sim(Ad, Bd, Cd, Dd, u[0])

# Design state feedback
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0.005], [0.1]])
poles = np.array([0.8, 0.85])
L = place(A, B, poles)
```

### Complete Documentation

For complete documentation of all functions with detailed descriptions, parameters, examples, and references, see **[mbrtc.md](./mbrtc.md)**.

## Project Files

- [demo_control_simulator.py](./demo_control_simulator.py): Multi-rate simulator for discrete-time control (Python 3.7 or higher, should be run in a terminal, see the comments!)
- [mbrtc.py](./mbrtc.py): Main library module with state-space analysis and control design functions
- [mbrtc.md](./mbrtc.md): Complete documentation for the mbrtc module
- [aliasing-practical.py](./aliasing-practical.py): Starting code for analyzing sampling and aliasing
- [notebooks](./notebooks): Series of Jupyter notebooks with example code numbered after the lectures
- [requirements.txt](./requirements.txt): Python package dependencies

## Author

Rufus Fraanje
Email: p.r.fraanje@hhs.nl

## License

GNU General Public License (GNU GPLv3)
