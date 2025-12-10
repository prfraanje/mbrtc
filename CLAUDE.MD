# CLAUDE.MD - Model-Based Real-Time Control (mbrtc)

## Project Overview

This repository contains course materials and Python code for Model-Based Real-Time Control, focusing on discrete-time state-space analysis and control design.

**Author:** Rufus Fraanje
**Email:** p.r.fraanje@hhs.nl
**License:** GNU General Public License (GNU GPLv3)

## Project Structure

### Core Files

- **mbrtc.py**: Main library providing functions for discrete-time state-space analysis and control design
  - Implements state-space models (continuous and discrete time)
  - Signal generation utilities (spike, step_signal, impulse_signal)
  - State-space conversion and simulation functions
  - Uses NumPy and SciPy (scipy.linalg, scipy.signal)

- **demo_control_simulator.py**: Multi-rate simulator for discrete-time control
  - Python 3.7 or higher required
  - Should be run in a terminal (see comments in file for usage)

- **aliasing-practical.py**: Starting code for analyzing sampling and aliasing effects

### Notebooks

The `notebooks/` directory contains a series of Jupyter notebooks (MBRTC1.ipynb through MBRTC7.ipynb) numbered after lectures, providing example code and demonstrations.

## Key Concepts & Symbols

The codebase uses the following notation (from mbrtc.py:12-37):

### Time and Signals
- `tc` - continuous time instants
- `td` - discrete time instants (1-D array)
- `h` - sampling time
- `u` - input signal (1-D array for single channel, 2-D array for multi-channel)
- `y` - output signal (same format as input)
- `x0` - initial state
- `x` - state sequence
- `NS` - number of samples
- `ni` - number of inputs
- `no` - number of outputs
- `ns` - number of states

### State-Space Models

**Continuous-time:** `Ac, Bc, Cc, Dc`
- `Ac` - state-transition matrix (ns × ns)
- `Bc` - input matrix (ns × ni)
- `Cc` - output matrix (no × ns)
- `Dc` - direct feedthrough matrix (no × ni)

**Discrete-time:** `Ad, Bd, Cd, Dd`
- `Ad` - state-transition matrix (Φ in Computer-Controlled Systems)
- `Bd` - input matrix (Γ in Computer-Controlled Systems)
- `Cd` - output matrix
- `Dd` - direct feedthrough matrix

## Dependencies

- Python 3.7 or higher
- NumPy
- SciPy (scipy.linalg, scipy.signal)
- Jupyter (for notebooks)

## Recent Changes

- Replaced obsolete `lsim2` function with `lsim`
- Corrected `sample_of_impulse` in `impulse_signal`
- Added convenience functions `step_signal` and `impulse_signal`

## Getting Started

1. Ensure Python 3.7+ is installed
2. Install dependencies: `pip install numpy scipy jupyter`
3. Import the main library: `from mbrtc import *`
4. Run the demo simulator in a terminal: `python demo_control_simulator.py`
5. Explore the numbered notebooks in sequence (MBRTC1.ipynb through MBRTC7.ipynb)

## Development Notes

- The project follows discrete-time control system conventions
- State-space models can be converted between continuous and discrete time
- Signal generation utilities support various test signals (spike, step, impulse)
- The simulator supports multi-rate discrete-time control scenarios
