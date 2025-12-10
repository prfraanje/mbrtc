# Model-Based Real-Time Control (mbrtc) - Documentation

## Module Overview

This module provides functions for analysis, simulation, and design of linear discrete-time and continuous-time control systems. It is designed for the course "Model-Based Real-Time Control" based on the textbook "Computer-Controlled Systems" by Åström and Wittenmark.

**Author:** Rufus Fraanje (p.r.fraanje@hhs.nl)
**License:** GNU General Public License (GNU GPLv3)
**Created:** 2019-03-06
**Last Modified:** 2025-12-10

---

## Table of Contents

### [Signal Generation](#signal-generation-1)
- [spike](#spike) - Generate discrete-time spike (unit impulse) signal
- [step_signal](#step_signal) - Generate discrete-time step signal
- [impulse_signal](#impulse_signal) - Generate discrete-time impulse with configurable amplitude
- [random_impulses](#random_impulses) - Generate random impulse signals for system identification

### [Continuous/Discrete Conversion](#continuousdiscrete-conversion-1)
- [c2d_zoh_AB](#c2d_zoh_ab) - Convert continuous (A,B) matrices to discrete (ZOH)
- [d2c_zoh_AB](#d2c_zoh_ab) - Convert discrete (A,B) matrices to continuous (inverse ZOH)
- [c2d_zoh](#c2d_zoh) - Convert continuous state-space (A,B,C,D) to discrete (ZOH)
- [d2c_zoh](#d2c_zoh) - Convert discrete state-space (A,B,C,D) to continuous
- [c2d_zoh_intersample](#c2d_zoh_intersample) - Convert to discrete with intersample output points
- [c2d_pole](#c2d_pole) - Convert poles from continuous to discrete time
- [d2c_pole](#d2c_pole) - Convert poles from discrete to continuous time
- [c2d_characteristic_equation](#c2d_characteristic_equation) - Convert characteristic polynomial c2d
- [d2c_characteristic_equation](#d2c_characteristic_equation) - Convert characteristic polynomial d2c

### [Simulation](#simulation-1)
- [sim_continuous](#sim_continuous) - Simulate continuous-time state-space model
- [sim_state](#sim_state) - Simulate discrete-time state evolution (no output)
- [sim](#sim) - Simulate discrete-time state-space model
- [sim_intersample](#sim_intersample) - Simulate with intersample output visualization

### [System Analysis](#system-analysis-1)
- [ctrb](#ctrb) - Compute controllability matrix
- [obsv](#obsv) - Compute observability matrix
- [is_reachable](#is_reachable) - Test if system is reachable/controllable
- [is_observable](#is_observable) - Test if system is observable
- [is_stable](#is_stable) - Test if system is stable (discrete or continuous)

### [Control Design](#control-design-1)
- [place](#place) - Compute state-feedback gain for pole placement
- [dc_gain](#dc_gain) - Compute DC (steady-state) gain

### [Canonical Forms & Transformations](#canonical-forms--transformations-1)
- [similarity_trans](#similarity_trans) - Apply similarity transformation to state-space matrices
- [canon_ctrl](#canon_ctrl) - Transform to controller canonical form
- [canon_obsv](#canon_obsv) - Transform to observer canonical form
- [canon](#canon) - Transform to canonical form (ctrl or obsv)

### [System Interconnections](#system-interconnections-1)
- [ss_series](#ss_series) - Series (cascade) interconnection of two systems
- [ss_parallel](#ss_parallel) - Parallel interconnection of two systems
- [ss_feedback](#ss_feedback) - Feedback interconnection of two systems

### [Utilities](#utilities-1)
- [pretty_polynomial](#pretty_polynomial) - Format polynomial coefficients as readable string

---

## Notation Conventions

### Time and Signals
- `tc, td` : Continuous/discrete time instants
- `h` : Sampling time
- `u, y` : Input/output signals (1-D or 2-D arrays)
- `x0, x` : Initial state / state sequence
- `NS` : Number of samples
- `ni, no, ns` : Number of inputs / outputs / states

### State-Space Matrices

**Continuous-time:** `Ac, Bc, Cc, Dc`
- `Ac` : State-transition matrix (A in Åström & Wittenmark)
- `Bc` : Input matrix (B in Åström & Wittenmark)
- `Cc` : Output matrix (C in Åström & Wittenmark)
- `Dc` : Feedthrough matrix (D in Åström & Wittenmark)

**Discrete-time:** `Ad, Bd, Cd, Dd`
- `Ad` : State-transition matrix (Φ in Åström & Wittenmark)
- `Bd` : Input matrix (Γ in Åström & Wittenmark)
- `Cd` : Output matrix (C in Åström & Wittenmark)
- `Dd` : Feedthrough matrix (D in Åström & Wittenmark)

---

## Quick Start Examples

```python
import numpy as np
from mbrtc import *

# Create a continuous-time first-order system and discretize
Ac = np.array([[-1.0]])
Bc = np.array([[1.0]])
Cc = np.array([[1.0]])
Dc = np.array([[0.0]])
h = 0.1
Ad, Bd, Cd, Dd = c2d_zoh(Ac, Bc, Cc, Dc, h)

# Simulate step response
u = step_signal(NS=50)
y = sim(Ad, Bd, Cd, Dd, u[0])

# Design state feedback for discrete system
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0.005], [0.1]])
poles = np.array([0.8, 0.85])
L = place(A, B, poles)
```

---

## Function Reference

## Signal Generation

### spike

```python
spike(NS=100, at_sample=1)
```

Generate a discrete-time spike (unit impulse) signal.

Creates a signal of zeros with a single spike (value of 1.0) at a specified sample index.

**Parameters:**
- `NS` : int, optional - Number of samples in the signal (default is 100)
- `at_sample` : int, optional - Sample index where the spike occurs (default is 1)

**Returns:**
- `signal` : ndarray, shape (1, NS) - Row vector containing the spike signal

**Example:**
```python
import matplotlib.pyplot as plt
spike_signal = spike(NS=20, at_sample=5)
plt.stem(spike_signal[0])
plt.title('Spike signal at sample 5')
```

**See Also:** [impulse_signal](#impulse_signal), [scipy.signal.unit_impulse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unit_impulse.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function)

---

### step_signal

```python
step_signal(NS=100, stepsize=1, sample_of_step=None)
```

Generate a discrete-time step signal.

Creates a step signal that transitions from 0 to a specified stepsize. If sample_of_step is None, the signal is constant at stepsize for all samples.

**Parameters:**
- `NS` : int, optional - Number of samples in the signal (default is 100)
- `stepsize` : float, optional - Amplitude of the step (default is 1)
- `sample_of_step` : int or None, optional - Sample index where the step occurs (default is None)

**Returns:**
- `signal` : ndarray, shape (1, NS) - Row vector containing the step signal

**Example:**
```python
import matplotlib.pyplot as plt
# Constant signal
constant = step_signal(NS=50, stepsize=2.5)
# Step at sample 10
step = step_signal(NS=50, stepsize=2.5, sample_of_step=10)
plt.plot(constant[0], label='Constant')
plt.plot(step[0], label='Step at sample 10')
plt.legend()
```

**See Also:** [impulse_signal](#impulse_signal), [scipy.signal.step](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: Step response](https://en.wikipedia.org/wiki/Step_response)

---

### impulse_signal

```python
impulse_signal(NS=100, impulsesize=1, sample_of_impulse=None)
```

Generate a discrete-time impulse signal with configurable amplitude.

Creates an impulse signal with a specified amplitude at a given sample index.

**Parameters:**
- `NS` : int, optional - Number of samples in the signal (default is 100)
- `impulsesize` : float, optional - Amplitude of the impulse (default is 1)
- `sample_of_impulse` : int or None, optional - Sample index where the impulse occurs (default is None, which means sample 0)

**Returns:**
- `signal` : ndarray, shape (1, NS) - Row vector containing the impulse signal

**Example:**
```python
import matplotlib.pyplot as plt
impulse = impulse_signal(NS=30, impulsesize=5.0, sample_of_impulse=10)
plt.stem(impulse[0])
plt.title('Impulse of amplitude 5.0 at sample 10')
```

**See Also:** [spike](#spike), [scipy.signal.impulse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: Impulse response](https://en.wikipedia.org/wiki/Impulse_response)

---

### random_impulses

```python
random_impulses(av_samples_per_spike=10, NS=100, nchan=1)
```

Generate random impulse signals for system identification.

Creates multi-channel random binary signals useful for exciting dynamic systems during identification experiments. Each channel independently generates random impulses with a specified average spacing.

**Parameters:**
- `av_samples_per_spike` : int, optional - Average number of samples between impulses (default is 10)
- `NS` : int, optional - Number of samples in the signal (default is 100)
- `nchan` : int, optional - Number of channels (default is 1)

**Returns:**
- `signal` : ndarray, shape (nchan, NS) - Matrix where each row is an independent random impulse signal

**Example:**
```python
import matplotlib.pyplot as plt
random_sig = random_impulses(av_samples_per_spike=5, NS=100, nchan=2)
plt.subplot(2, 1, 1)
plt.stem(random_sig[0])
plt.title('Channel 1')
plt.subplot(2, 1, 2)
plt.stem(random_sig[1])
plt.title('Channel 2')
```

**See Also:** [impulse_signal](#impulse_signal), [numpy.random.randint](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 11
- [Wikipedia: System identification](https://en.wikipedia.org/wiki/System_identification)

---

## Continuous/Discrete Conversion

### c2d_zoh_AB

```python
c2d_zoh_AB(Ac, Bc, h)
```

Convert continuous-time state matrices to discrete-time using zero-order hold.

Discretizes the state-transition matrix Ac and input matrix Bc assuming zero-order hold on the input. Uses matrix exponential for exact discretization.

**Parameters:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix
- `h` : float - Sampling time

**Returns:**
- `Ad` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix (Φ in Åström & Wittenmark)
- `Bd` : ndarray, shape (ns, ni) - Discrete-time input matrix (Γ in Åström & Wittenmark)

**Notes:**

The conversion is computed using:
- Ad = exp(Ac * h)
- Bd = ∫[0,h] exp(Ac * τ) dτ * Bc

This is implemented efficiently using the combined matrix exponential method.

**Example:**
```python
# Continuous-time double integrator
Ac = np.array([[0, 1], [0, 0]])
Bc = np.array([[0], [1]])
h = 0.1
Ad, Bd = c2d_zoh_AB(Ac, Bc, h)
```

**See Also:** [c2d_zoh](#c2d_zoh), [d2c_zoh_AB](#d2c_zoh_ab), [scipy.signal.cont2discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: Discretization of linear state space models](https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models)

---

### d2c_zoh_AB

```python
d2c_zoh_AB(Ad, Bd, h)
```

Convert discrete-time state matrices to continuous-time (inverse of c2d_zoh_AB).

Computes the continuous-time state-transition and input matrices from their discrete-time equivalents, assuming zero-order hold was used for discretization. Uses matrix logarithm for the inverse transformation.

**Parameters:**
- `Ad` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix (Φ in Åström & Wittenmark)
- `Bd` : ndarray, shape (ns, ni) - Discrete-time input matrix (Γ in Åström & Wittenmark)
- `h` : float - Sampling time

**Returns:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix

**Notes:**

This is the inverse operation of c2d_zoh_AB, using matrix logarithm. The conversion may be numerically sensitive for some systems.

**Example:**
```python
Ad = np.array([[1, 0.1], [0, 1]])
Bd = np.array([[0.005], [0.1]])
h = 0.1
Ac, Bc = d2c_zoh_AB(Ad, Bd, h)
```

**See Also:** [d2c_zoh](#d2c_zoh), [c2d_zoh_AB](#c2d_zoh_ab), [scipy.linalg.logm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2

---

### c2d_zoh

```python
c2d_zoh(Ac, Bc, Cc, Dc, h)
```

Convert continuous-time state-space model to discrete-time using zero-order hold.

Discretizes a complete state-space model (A, B, C, D) assuming zero-order hold on the input. The output and feedthrough matrices remain unchanged (Cd = Cc, Dd = Dc).

**Parameters:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix
- `Cc` : ndarray, shape (no, ns) - Continuous-time output matrix
- `Dc` : ndarray, shape (no, ni) - Continuous-time feedthrough matrix
- `h` : float - Sampling time

**Returns:**
- `Ad` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix
- `Bd` : ndarray, shape (ns, ni) - Discrete-time input matrix
- `Cd` : ndarray, shape (no, ns) - Discrete-time output matrix (copy of Cc)
- `Dd` : ndarray, shape (no, ni) - Discrete-time feedthrough matrix (copy of Dc)

**Example:**
```python
# First-order system
Ac = np.array([[-1.0]])
Bc = np.array([[1.0]])
Cc = np.array([[1.0]])
Dc = np.array([[0.0]])
h = 0.1
Ad, Bd, Cd, Dd = c2d_zoh(Ac, Bc, Cc, Dc, h)
```

**See Also:** [c2d_zoh_AB](#c2d_zoh_ab), [d2c_zoh](#d2c_zoh), [scipy.signal.cont2discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: Discretization of linear state space models](https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models)

---

### d2c_zoh

```python
d2c_zoh(Ad, Bd, Cd, Dd, h)
```

Convert discrete-time state-space model to continuous-time (inverse of c2d_zoh).

Computes the continuous-time state-space model from its discrete-time equivalent, assuming zero-order hold was used for discretization. The output and feedthrough matrices remain unchanged (Cc = Cd, Dc = Dd).

**Parameters:**
- `Ad` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix
- `Bd` : ndarray, shape (ns, ni) - Discrete-time input matrix
- `Cd` : ndarray, shape (no, ns) - Discrete-time output matrix
- `Dd` : ndarray, shape (no, ni) - Discrete-time feedthrough matrix
- `h` : float - Sampling time

**Returns:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix
- `Cc` : ndarray, shape (no, ns) - Continuous-time output matrix (copy of Cd)
- `Dc` : ndarray, shape (no, ni) - Continuous-time feedthrough matrix (copy of Dd)

**Example:**
```python
Ad = np.array([[0.9048]])
Bd = np.array([[0.0952]])
Cd = np.array([[1.0]])
Dd = np.array([[0.0]])
h = 0.1
Ac, Bc, Cc, Dc = d2c_zoh(Ad, Bd, Cd, Dd, h)
```

**See Also:** [d2c_zoh_AB](#d2c_zoh_ab), [c2d_zoh](#c2d_zoh)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2

---

### c2d_zoh_intersample

```python
c2d_zoh_intersample(Ac, Bc, Cc, Dc, h, number_intersamples)
```

Convert continuous-time model to discrete-time with intersample outputs.

Creates a discrete-time model that can compute outputs at multiple time points within each sampling interval. This is useful for visualizing the continuous-time behavior between samples when simulating with zero-order hold inputs.

**Parameters:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix
- `Cc` : ndarray, shape (no, ns) - Continuous-time output matrix
- `Dc` : ndarray, shape (no, ni) - Continuous-time feedthrough matrix
- `h` : float - Sampling time
- `number_intersamples` : int - Number of output time points per sampling interval

**Returns:**
- `Ad` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix
- `Bd` : ndarray, shape (ns, ni) - Discrete-time input matrix
- `Cd` : ndarray, shape (number_intersamples*no, ns) - Extended output matrix for intersample outputs
- `Dd` : ndarray, shape (number_intersamples*no, ni) - Extended feedthrough matrix for intersample outputs

**Notes:**

The extended output matrices Cd and Dd are stacked vertically, with each block corresponding to an intersample time point. Used internally by sim_intersample for high-resolution visualization of continuous-time behavior.

**Example:**
```python
Ac = np.array([[-1.0]])
Bc = np.array([[1.0]])
Cc = np.array([[1.0]])
Dc = np.array([[0.0]])
h = 0.1
Ad, Bd, Cd, Dd = c2d_zoh_intersample(Ac, Bc, Cc, Dc, h, 10)
# Cd will have shape (10, 1) for 10 intersample points
```

**See Also:** [sim_intersample](#sim_intersample), [c2d_zoh](#c2d_zoh)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)

---

### c2d_pole

```python
c2d_pole(lambda_i, h)
```

Convert poles from continuous-time to discrete-time.

Converts continuous-time pole locations (eigenvalues) to their discrete-time equivalents using the mapping z = exp(λ*h).

**Parameters:**
- `lambda_i` : complex or ndarray - Continuous-time pole(s) (eigenvalue(s))
- `h` : float - Sampling time

**Returns:**
- `z_i` : complex or ndarray - Discrete-time pole(s), same shape as lambda_i

**Notes:**

The transformation follows: z = exp(λ*h)
- Continuous-time stable poles (Re(λ) < 0) map to stable discrete-time poles (|z| < 1)
- The imaginary axis (Re(λ) = 0) maps to the unit circle (|z| = 1)

**Example:**
```python
# Stable continuous-time pole
lambda_c = -1.0 + 2.0j
h = 0.1
z = c2d_pole(lambda_c, h)
print(f"Continuous pole: {lambda_c}")
print(f"Discrete pole: {z}, magnitude: {abs(z):.3f}")
```

**See Also:** [d2c_pole](#d2c_pole), [c2d_zoh](#c2d_zoh)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: Bilinear transform](https://en.wikipedia.org/wiki/Bilinear_transform)

---

### d2c_pole

```python
d2c_pole(lambda_i, h)
```

Convert poles from discrete-time to continuous-time.

Converts discrete-time pole locations to their continuous-time equivalents using the inverse mapping λ = ln(z)/h.

**Parameters:**
- `lambda_i` : complex or ndarray - Discrete-time pole(s) (z-domain eigenvalue(s))
- `h` : float - Sampling time

**Returns:**
- `s_i` : complex or ndarray - Continuous-time pole(s), same shape as lambda_i

**Notes:**

The transformation follows: λ = ln(z)/h
- Discrete-time stable poles (|z| < 1) map to stable continuous-time poles (Re(λ) < 0)
- The unit circle (|z| = 1) maps to the imaginary axis (Re(λ) = 0)
- This is the inverse of c2d_pole

**Example:**
```python
# Stable discrete-time pole
z = 0.9 + 0.1j
h = 0.1
lambda_c = d2c_pole(z, h)
print(f"Discrete pole: {z}, magnitude: {abs(z):.3f}")
print(f"Continuous pole: {lambda_c}")
```

**See Also:** [c2d_pole](#c2d_pole), [d2c_zoh](#d2c_zoh), [numpy.log](https://numpy.org/doc/stable/reference/generated/numpy.log.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: Bilinear transform](https://en.wikipedia.org/wiki/Bilinear_transform)

---

### c2d_characteristic_equation

```python
c2d_characteristic_equation(ac, h)
```

Convert characteristic polynomial from continuous to discrete-time.

Converts the characteristic polynomial of a continuous-time system to its discrete-time equivalent using the zero-order hold assumption.

**Parameters:**
- `ac` : ndarray, shape (n+1,) - Coefficients of continuous-time characteristic polynomial in descending order
- `h` : float - Sampling time

**Returns:**
- `ad` : ndarray, shape (n+1,) - Coefficients of discrete-time characteristic polynomial in descending order

**Notes:**

The conversion is performed by:
1. Constructing a companion matrix from the continuous polynomial
2. Discretizing using matrix exponential
3. Computing the characteristic polynomial of the discrete matrix

**Example:**
```python
# Second-order continuous characteristic equation: s^2 + 2s + 1
ac = np.array([1, 2, 1])
h = 0.1
ad = c2d_characteristic_equation(ac, h)
print(f"Discrete characteristic polynomial: {ad}")
```

**See Also:** [d2c_characteristic_equation](#d2c_characteristic_equation), [c2d_pole](#c2d_pole), [numpy.poly](https://numpy.org/doc/stable/reference/generated/numpy.poly.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: Characteristic polynomial](https://en.wikipedia.org/wiki/Characteristic_polynomial)

---

### d2c_characteristic_equation

```python
d2c_characteristic_equation(ad, h)
```

Convert characteristic polynomial from discrete-time to continuous-time.

Converts the characteristic polynomial of a discrete-time system to its continuous-time equivalent (inverse of c2d_characteristic_equation).

**Parameters:**
- `ad` : ndarray, shape (n+1,) - Coefficients of discrete-time characteristic polynomial in descending order
- `h` : float - Sampling time

**Returns:**
- `ac` : ndarray, shape (n+1,) - Coefficients of continuous-time characteristic polynomial in descending order

**Notes:**

The conversion is performed by:
1. Constructing a companion matrix from the discrete polynomial
2. Computing continuous equivalent using matrix logarithm
3. Computing the characteristic polynomial of the continuous matrix

This is the inverse operation of c2d_characteristic_equation.

**Example:**
```python
# Discrete characteristic polynomial
ad = np.array([1, -1.8, 0.81])
h = 0.1
ac = d2c_characteristic_equation(ad, h)
print(f"Continuous characteristic polynomial: {ac}")
```

**See Also:** [c2d_characteristic_equation](#c2d_characteristic_equation), [d2c_pole](#d2c_pole), [scipy.linalg.logm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: Characteristic polynomial](https://en.wikipedia.org/wiki/Characteristic_polynomial)

---

## Simulation

### sim_continuous

```python
sim_continuous(A, B, C, D, u=None, t=None, x0=None, return_X=False)
```

Simulate continuous-time linear state-space model.

Wrapper around scipy.signal.lsim for simulating continuous-time LTI systems. Computes the output response to an input signal.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, ni) - Input matrix
- `C` : ndarray, shape (no, ns) - Output matrix
- `D` : ndarray, shape (no, ni) - Feedthrough matrix
- `u` : ndarray, optional - Input signal array
- `t` : ndarray, optional - Time vector, shape (N,)
- `x0` : ndarray, optional - Initial state vector, shape (ns,)
- `return_X` : bool, optional - If True, also return state trajectory (default is False)

**Returns:**
- `T` : ndarray, shape (N,) - Time vector
- `yout` : ndarray, shape (N,) or (N, no) - Output response
- `xout` : ndarray, shape (N, ns), optional - State trajectory (only if return_X=True)

**Example:**
```python
# First-order system
A = np.array([[-1.0]])
B = np.array([[1.0]])
C = np.array([[1.0]])
D = np.array([[0.0]])
t = np.linspace(0, 5, 100)
u = np.ones_like(t)
T, y = sim_continuous(A, B, C, D, u=u, t=t)
import matplotlib.pyplot as plt
plt.plot(T, y)
plt.xlabel('Time')
plt.ylabel('Output')
```

**See Also:** [sim](#sim), [scipy.signal.lsim](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)

---

### sim_state

```python
sim_state(A, B, u, x0=None)
```

Simulate discrete-time state evolution (without output equation).

Computes the state trajectory of a discrete-time system given input sequence. Iterates the state equation: x[k+1] = A*x[k] + B*u[k].

**Parameters:**
- `A` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix
- `B` : ndarray, shape (ns, ni) - Discrete-time input matrix
- `u` : ndarray - Input signal
- `x0` : ndarray, optional - Initial state vector, shape (ns,)

**Returns:**
- `x` : ndarray, shape (ns, N) or (N,) - State trajectory

**Notes:**

This function only computes the state evolution, not the output. Use sim() to compute both state and output trajectories.

**Example:**
```python
# Double integrator system
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0.005], [0.1]])
u = np.ones((1, 50))  # Step input
x0 = np.array([0, 0])
x = sim_state(A, B, u, x0)
import matplotlib.pyplot as plt
plt.plot(x[0,:], label='Position')
plt.plot(x[1,:], label='Velocity')
plt.legend()
```

**See Also:** [sim](#sim), [sim_continuous](#sim_continuous)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: State-space representation](https://en.wikipedia.org/wiki/State-space_representation)

---

### sim

```python
sim(A, B, C, D, u, x0=None, return_X=False)
```

Simulate discrete-time linear state-space model.

Computes the output response of a discrete-time system to an input sequence. Iterates: x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k].

**Parameters:**
- `A` : ndarray, shape (ns, ns) - Discrete-time state-transition matrix
- `B` : ndarray, shape (ns, ni) - Discrete-time input matrix
- `C` : ndarray, shape (no, ns) - Discrete-time output matrix
- `D` : ndarray, shape (no, ni) - Discrete-time feedthrough matrix
- `u` : ndarray - Input signal
- `x0` : ndarray, optional - Initial state vector, shape (ns,)
- `return_X` : bool, optional - If True, also return state trajectory (default is False)

**Returns:**
- `y` : ndarray, shape (N,) or (no, N) - Output response
- `x` : ndarray, shape (ns, N) or (N,), optional - State trajectory (only if return_X=True)

**Notes:**

Input u should be organized with time along columns (each column is one time step). This is consistent with discrete-time control conventions.

**Example:**
```python
# First-order discrete system
A = np.array([[0.9]])
B = np.array([[0.1]])
C = np.array([[1.0]])
D = np.array([[0.0]])
u = step_signal(NS=50)
y = sim(A, B, C, D, u[0])
import matplotlib.pyplot as plt
plt.plot(y)
plt.title('Step response')
```

**See Also:** [sim_state](#sim_state), [sim_continuous](#sim_continuous), [sim_intersample](#sim_intersample)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: State-space representation](https://en.wikipedia.org/wiki/State-space_representation)

---

### sim_intersample

```python
sim_intersample(Ac, Bc, Cc, Dc, h, number_is, ud, td, x0=None)
```

Simulate continuous-time system with intersample output visualization.

Simulates a continuous-time system discretized with zero-order hold, but computes outputs at multiple points within each sampling interval. This provides high-resolution visualization of the continuous-time behavior between sample points.

**Parameters:**
- `Ac` : ndarray, shape (ns, ns) - Continuous-time state-transition matrix
- `Bc` : ndarray, shape (ns, ni) - Continuous-time input matrix
- `Cc` : ndarray, shape (no, ns) - Continuous-time output matrix
- `Dc` : ndarray, shape (no, ni) - Continuous-time feedthrough matrix
- `h` : float - Sampling time
- `number_is` : int - Number of (equidistant) intersample points per sampling interval
- `ud` : ndarray, shape (ni, NS) or (NS,) - Discrete-time zero-order hold input signal
- `td` : ndarray, shape (NS,) - Discrete-time vector (sampling instants)
- `x0` : ndarray, optional - Initial state vector, shape (ns,)

**Returns:**
- `td_is` : ndarray, shape (NS*number_is,) - Time vector including all intersample points
- `yd_is` : ndarray, shape (NS*number_is,) - Output at all intersample points

**Notes:**

This function is useful for visualizing the ripple and intersample behavior that occurs in sampled-data systems, especially important when analyzing anti-aliasing effects and output response between samples.

**Example:**
```python
# First-order continuous system
Ac = np.array([[-1.0]])
Bc = np.array([[1.0]])
Cc = np.array([[1.0]])
Dc = np.array([[0.0]])
h = 0.1
NS = 50
td = np.arange(NS) * h
ud = step_signal(NS=NS)
td_is, yd_is = sim_intersample(Ac, Bc, Cc, Dc, h, 10, ud, td)
import matplotlib.pyplot as plt
plt.plot(td_is, yd_is, '-', label='Intersample output')
plt.plot(td, yd_is[::10], 'o', label='Sample points')
plt.legend()
```

**See Also:** [sim](#sim), [c2d_zoh_intersample](#c2d_zoh_intersample), [sim_continuous](#sim_continuous)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 3
- [Wikipedia: Intersample behavior](https://en.wikipedia.org/wiki/Intersample_behavior)

---

## System Analysis

### ctrb

```python
ctrb(A, B)
```

Compute the controllability matrix of a state-space system.

Returns the controllability matrix Wc = [B AB A²B ... A^(n-1)B]. The system is controllable if and only if Wc has full row rank.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, ni) - Input matrix

**Returns:**
- `Wc` : ndarray, shape (ns, ns*ni) - Controllability matrix

**Notes:**

The controllability matrix is used to test if all states can be reached from any initial state using appropriate inputs. The system (A, B) is controllable if rank(Wc) = ns.

**Example:**
```python
# Double integrator (controllable)
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
Wc = ctrb(A, B)
print(f"Rank: {np.linalg.matrix_rank(Wc)}")  # Should be 2
```

**See Also:** [obsv](#obsv), [is_reachable](#is_reachable), [numpy.linalg.matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Controllability](https://en.wikipedia.org/wiki/Controllability)
- [MATLAB: ctrb](https://www.mathworks.com/help/control/ref/lti.ctrb.html)

---

### obsv

```python
obsv(A, C)
```

Compute the observability matrix of a state-space system.

Returns the observability matrix Wo = [C; CA; CA²; ...; CA^(n-1)]. The system is observable if and only if Wo has full column rank.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `C` : ndarray, shape (no, ns) - Output matrix

**Returns:**
- `Wo` : ndarray, shape (ns*no, ns) - Observability matrix (vertically stacked)

**Notes:**

The observability matrix is used to test if the internal state can be determined from output measurements. The system (A, C) is observable if rank(Wo) = ns.

**Example:**
```python
# Double integrator with position measurement (observable)
A = np.array([[0, 1], [0, 0]])
C = np.array([[1, 0]])
Wo = obsv(A, C)
print(f"Rank: {np.linalg.matrix_rank(Wo)}")  # Should be 2
```

**See Also:** [ctrb](#ctrb), [is_observable](#is_observable), [numpy.linalg.matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Observability](https://en.wikipedia.org/wiki/Observability)
- [MATLAB: obsv](https://www.mathworks.com/help/control/ref/lti.obsv.html)

---

### is_reachable

```python
is_reachable(A, B)
```

Test if state-space system is reachable (controllable).

Determines whether all states can be reached from any initial state using appropriate control inputs.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, ni) - Input matrix

**Returns:**
- `reachable` : bool - True if the system is reachable, False otherwise

**Notes:**

A system is reachable if rank(controllability_matrix) = ns. For linear systems, reachability and controllability are equivalent.

**Example:**
```python
# Double integrator (reachable)
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
print(is_reachable(A, B))  # True

# Not reachable system
A = np.array([[1, 0], [0, 2]])
B = np.array([[1], [0]])
print(is_reachable(A, B))  # False
```

**See Also:** [ctrb](#ctrb), [is_observable](#is_observable), [place](#place)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Controllability](https://en.wikipedia.org/wiki/Controllability)

---

### is_observable

```python
is_observable(A, C)
```

Test if state-space system is observable.

Determines whether the internal state can be determined from output measurements over time.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `C` : ndarray, shape (no, ns) - Output matrix

**Returns:**
- `observable` : bool - True if the system is observable, False otherwise

**Notes:**

A system is observable if rank(observability_matrix) = ns. Observability is the dual property to controllability.

**Example:**
```python
# Double integrator with position measurement (observable)
A = np.array([[0, 1], [0, 0]])
C = np.array([[1, 0]])
print(is_observable(A, C))  # True

# Not observable system
A = np.array([[1, 0], [0, 2]])
C = np.array([[1, 0]])
print(is_observable(A, C))  # False
```

**See Also:** [obsv](#obsv), [is_reachable](#is_reachable), [canon_obsv](#canon_obsv)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Observability](https://en.wikipedia.org/wiki/Observability)

---

### is_stable

```python
is_stable(A, domain="discrete")
```

Test if state-transition matrix is stable.

Checks stability by examining eigenvalue locations. For discrete-time systems, all eigenvalues must lie inside the unit circle. For continuous-time systems, all eigenvalues must have negative real parts.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `domain` : str, optional - Time domain: "discrete" or "d" for discrete-time (default), "continuous" or "c" for continuous-time

**Returns:**
- `stable` : bool - True if the system is stable, False otherwise

**Notes:**

- Discrete-time: stable if all |λ_i| < 1 (inside unit circle)
- Continuous-time: stable if all Re(λ_i) < 0 (left half-plane)

**Example:**
```python
# Stable discrete-time system
A_d = np.array([[0.9, 0.1], [0, 0.8]])
print(is_stable(A_d, domain="discrete"))  # True

# Stable continuous-time system
A_c = np.array([[-1, 1], [0, -2]])
print(is_stable(A_c, domain="continuous"))  # True

# Unstable discrete-time system
A_unstable = np.array([[1.1, 0], [0, 0.9]])
print(is_stable(A_unstable, domain="d"))  # False
```

**See Also:** [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html), [c2d_pole](#c2d_pole), [place](#place)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 2
- [Wikipedia: Stability theory](https://en.wikipedia.org/wiki/Stability_theory)
- [Wikipedia: Routh–Hurwitz stability criterion](https://en.wikipedia.org/wiki/Routh%E2%80%93Hurwitz_stability_criterion)

---

## Control Design

### place

```python
place(A, B, pole_vec)
```

Compute state-feedback gain matrix for pole placement.

Calculates the state-feedback gain matrix L such that the closed-loop system u = -L*x has poles at the specified locations. Uses the controller canonical form method for single-input systems.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix (open-loop)
- `B` : ndarray, shape (ns, 1) - Input matrix (single-input only)
- `pole_vec` : ndarray, shape (ns,) - Desired closed-loop pole locations

**Returns:**
- `L` : ndarray, shape (1, ns) - State-feedback gain matrix. Use control law: u = -L*x

**Notes:**

The closed-loop system with u = -L*x has dynamics: x[k+1] = (A - B*L)*x[k]

The system must be reachable for arbitrary pole placement. Pole locations should be chosen for stability and performance.

- For discrete-time systems: place poles inside unit circle (|z| < 1)
- For continuous-time systems: place poles in left half-plane (Re(s) < 0)

**Example:**
```python
# Discrete-time double integrator
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0.005], [0.1]])
# Desired poles at z = 0.8 ± 0.1j
poles = np.array([0.8 + 0.1j, 0.8 - 0.1j])
L = place(A, B, poles)
# Verify closed-loop poles
A_cl = A - B @ L
poles_cl = np.linalg.eig(A_cl)[0]
print(f"Closed-loop poles: {poles_cl}")
```

**See Also:** [canon_ctrl](#canon_ctrl), [is_reachable](#is_reachable), [is_stable](#is_stable), [scipy.signal.place_poles](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 5
- [Wikipedia: Pole placement](https://en.wikipedia.org/wiki/Pole_placement)
- [MATLAB: place](https://www.mathworks.com/help/control/ref/place.html)

---

### dc_gain

```python
dc_gain(A, B, C, D, domain="discrete")
```

Compute DC (steady-state) gain of a state-space system.

Calculates the steady-state gain from input to output when applying a constant (step) input to the system.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, ni) - Input matrix
- `C` : ndarray, shape (no, ns) - Output matrix
- `D` : ndarray, shape (no, ni) - Feedthrough matrix
- `domain` : str, optional - Time domain: "discrete" or "d" for discrete-time (default), "continuous" or "c" for continuous-time

**Returns:**
- `gain` : ndarray, shape (no, ni) - DC gain matrix. For SISO systems, this is a scalar

**Notes:**

- Discrete-time: DC gain = C*(I - A)^(-1)*B + D, evaluated at z = 1
- Continuous-time: DC gain = -C*A^(-1)*B + D, evaluated at s = 0

The system must be stable for the DC gain to be meaningful.

**Example:**
```python
# Discrete-time first-order system
A = np.array([[0.9]])
B = np.array([[0.1]])
C = np.array([[1.0]])
D = np.array([[0.0]])
gain = dc_gain(A, B, C, D, domain="discrete")
print(f"DC gain: {gain}")  # Should be 1.0 for this system
```

**See Also:** [is_stable](#is_stable), [scipy.signal.dcgain](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dcgain.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [Wikipedia: DC bias](https://en.wikipedia.org/wiki/DC_bias)
- [MATLAB: dcgain](https://www.mathworks.com/help/control/ref/lti.dcgain.html)

---

## Canonical Forms & Transformations

### similarity_trans

```python
similarity_trans(A, B, C, T)
```

Apply similarity transformation to state-space matrices.

Transforms a state-space system to a new coordinate system defined by the transformation matrix T, where x_transformed = T * x.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - Original state-transition matrix
- `B` : ndarray, shape (ns, ni) - Original input matrix
- `C` : ndarray, shape (no, ns) - Original output matrix
- `T` : ndarray, shape (ns, ns) - Transformation matrix (must be non-singular)

**Returns:**
- `At` : ndarray, shape (ns, ns) - Transformed state-transition matrix: At = T * A * T^(-1)
- `Bt` : ndarray, shape (ns, ni) - Transformed input matrix: Bt = T * B
- `Ct` : ndarray, shape (no, ns) - Transformed output matrix: Ct = C * T^(-1)

**Notes:**

The feedthrough matrix D remains unchanged under similarity transformation. The transformation preserves input-output behavior, eigenvalues, and transfer function. Used for canonical forms and numerical conditioning.

**Example:**
```python
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
T = np.array([[1, 1], [1, 2]])
At, Bt, Ct = similarity_trans(A, B, C, T)
```

**See Also:** [canon_ctrl](#canon_ctrl), [canon_obsv](#canon_obsv), [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Canonical realizations](https://en.wikipedia.org/wiki/State-space_representation#Canonical_realizations)

---

### canon_ctrl

```python
canon_ctrl(A, B)
```

Compute transformation matrix to controller canonical form.

Finds the similarity transformation T that converts a state-space system to controller canonical form (also called controllable canonical form), where state-feedback gain calculation is straightforward.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, 1) - Input matrix (single-input only)

**Returns:**
- `Tctrl` : ndarray, shape (ns, ns) - Transformation matrix to controller canonical form

**Raises:**
- Exception - If the system (A, B) is not reachable
- NotImplementedError - If the system has multiple inputs (ni > 1)

**Notes:**

Controller canonical form has a specific structure where the last row of A contains the negative coefficients of the characteristic polynomial. This form is useful for state-feedback design using pole placement.

The transformation satisfies:
- A_ctrl = T * A * T^(-1)
- B_ctrl = T * B = [1, 0, ..., 0]^T

**Example:**
```python
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
T = canon_ctrl(A, B)
# Use with similarity_trans to get canonical form
C = np.array([[1, 0]])
At, Bt, Ct = similarity_trans(A, B, C, T)
```

**See Also:** [canon_obsv](#canon_obsv), [canon](#canon), [similarity_trans](#similarity_trans), [place](#place)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- Ding, F. (2010). Transformations between some special matrices. Computers and Mathematics with Applications, 59, 2676-2694
- [Wikipedia: Canonical form (control theory)](https://en.wikipedia.org/wiki/Canonical_form#Control_theory)

---

### canon_obsv

```python
canon_obsv(A, C)
```

Compute transformation matrix to observer canonical form.

Finds the similarity transformation T that converts a state-space system to observer canonical form (also called observable canonical form), which is useful for state observer/estimator design.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `C` : ndarray, shape (1, ns) - Output matrix (single-output only)

**Returns:**
- `Tobsv` : ndarray, shape (ns, ns) - Transformation matrix to observer canonical form

**Raises:**
- Exception - If the system (A, C) is not observable
- NotImplementedError - If the system has multiple outputs (no > 1)

**Notes:**

Observer canonical form is the dual of controller canonical form. The first column of A contains the negative coefficients of the characteristic polynomial. This form is useful for observer design.

The transformation satisfies:
- A_obsv = T * A * T^(-1)
- C_obsv = C * T^(-1) = [1, 0, ..., 0]

**Example:**
```python
A = np.array([[0, 1], [-2, -3]])
C = np.array([[1, 0]])
T = canon_obsv(A, C)
# Use with similarity_trans to get canonical form
B = np.array([[0], [1]])
At, Bt, Ct = similarity_trans(A, B, C, T)
```

**See Also:** [canon_ctrl](#canon_ctrl), [canon](#canon), [similarity_trans](#similarity_trans), [obsv](#obsv)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4
- [Wikipedia: Canonical form (control theory)](https://en.wikipedia.org/wiki/Canonical_form#Control_theory)

---

### canon

```python
canon(A, B, C, method='ctrl')
```

Transform state-space system to canonical form.

Convenience function that transforms a system to either controller or observer canonical form.

**Parameters:**
- `A` : ndarray, shape (ns, ns) - State-transition matrix
- `B` : ndarray, shape (ns, ni) - Input matrix
- `C` : ndarray, shape (no, ns) - Output matrix
- `method` : str, optional - Canonical form to use: 'ctrl' for controller canonical form (default), 'obsv' for observer canonical form

**Returns:**
- `At` : ndarray, shape (ns, ns) - Transformed state-transition matrix in canonical form
- `Bt` : ndarray, shape (ns, ni) - Transformed input matrix in canonical form
- `Ct` : ndarray, shape (no, ns) - Transformed output matrix in canonical form

**Raises:**
- NotImplementedError - If method is not 'ctrl' or 'obsv'

**Example:**
```python
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
At, Bt, Ct = canon(A, B, C, method='ctrl')
print("Controller canonical form A:\\n", At)
```

**See Also:** [canon_ctrl](#canon_ctrl), [canon_obsv](#canon_obsv), [similarity_trans](#similarity_trans)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 4

---

## System Interconnections

### ss_series

```python
ss_series(A1, B1, C1, D1, A2, B2, C2, D2)
```

Compute series (cascade) interconnection of two state-space systems.

Connects two systems in series where the output of the first system becomes the input to the second system: u --> System1 --> System2 --> y.

**Parameters:**
- `A1, B1, C1, D1` : ndarray - State-space matrices of first system (upstream)
- `A2, B2, C2, D2` : ndarray - State-space matrices of second system (downstream)

**Returns:**
- `A, B, C, D` : ndarray - State-space matrices of the series interconnection

**Notes:**

The state vector of the series connection is [x1; x2] where x1 and x2 are the states of the first and second systems respectively.

The transfer function of the series connection is G(z) = G2(z) * G1(z), meaning the output of System1 feeds into System2.

Interconnection diagram:
```
input --> [System1] --> [System2] --> output
```

**Example:**
```python
# First system: gain of 2
A1 = np.array([[0.5]])
B1 = np.array([[1.0]])
C1 = np.array([[2.0]])
D1 = np.array([[0.0]])
# Second system: delay
A2 = np.array([[0]])
B2 = np.array([[1]])
C2 = np.array([[1]])
D2 = np.array([[0]])
A, B, C, D = ss_series(A1, B1, C1, D1, A2, B2, C2, D2)
```

**See Also:** [ss_parallel](#ss_parallel), [ss_feedback](#ss_feedback)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [StackExchange: LTI interconnections](https://math.stackexchange.com/questions/2933407/formula-for-interconnection-of-ltis-in-state-space-form)

---

### ss_parallel

```python
ss_parallel(A1, B1, C1, D1, A2, B2, C2, D2, sign=None)
```

Compute parallel interconnection of two state-space systems.

Connects two systems in parallel where both receive the same input and their outputs are summed: y = y1 + sign*y2.

**Parameters:**
- `A1, B1, C1, D1` : ndarray - State-space matrices of first system
- `A2, B2, C2, D2` : ndarray - State-space matrices of second system
- `sign` : ndarray or None, optional - Sign vector for each output channel. If None, all signs are positive (simple addition)

**Returns:**
- `A, B, C, D` : ndarray - State-space matrices of the parallel interconnection

**Notes:**

The state vector of the parallel connection is [x1; x2] where x1 and x2 are the states of the first and second systems respectively.

The transfer function of the parallel connection is: G(z) = G1(z) + sign*G2(z)

Interconnection diagram:
```
                          +
input --> [System1] -->Σ--> output
  |                    ^sign
  └-----> [System2] ---|
```

**Example:**
```python
# Sum two first-order systems
A1 = np.array([[0.9]])
B1 = np.array([[0.1]])
C1 = np.array([[1.0]])
D1 = np.array([[0.0]])
A2 = np.array([[0.8]])
B2 = np.array([[0.2]])
C2 = np.array([[1.0]])
D2 = np.array([[0.0]])
# Positive sum
A, B, C, D = ss_parallel(A1, B1, C1, D1, A2, B2, C2, D2)
# Difference (negative sign)
A, B, C, D = ss_parallel(A1, B1, C1, D1, A2, B2, C2, D2, sign=np.array([-1]))
```

**See Also:** [ss_series](#ss_series), [ss_feedback](#ss_feedback)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)
- [StackExchange: LTI interconnections](https://math.stackexchange.com/questions/2933407/formula-for-interconnection-of-ltis-in-state-space-form)

---

### ss_feedback

```python
ss_feedback(A1, B1, C1, D1, A2, B2, C2, D2, sign=None)
```

Compute feedback interconnection of two state-space systems.

Connects two systems in a feedback loop where the output of the first system is the overall output, and the second system is in the feedback path.

**Parameters:**
- `A1, B1, C1, D1` : ndarray - State-space matrices of forward path system
- `A2, B2, C2, D2` : ndarray - State-space matrices of feedback path system
- `sign` : ndarray or None, optional - Sign vector for each feedback channel. If None, all signs are negative (negative feedback)

**Returns:**
- `Acl, Bcl, Ccl, Dcl` : ndarray - State-space matrices of the closed-loop system

**Notes:**

The state vector of the feedback connection is [x1; x2] where x1 and x2 are the states of the forward and feedback systems respectively.

Default is negative feedback (sign = -1), which is standard in control systems.

Interconnection diagram:
```
input --> Σ --> [System1] --> output
  sign    ^           |
          |           v
          +--- [System2]
```

The closed-loop transfer function is: G_cl(z) = G1(z) / (1 - sign*G1(z)*G2(z))

**Example:**
```python
# Unity feedback control system
# Plant: first-order system
A1 = np.array([[0.9]])
B1 = np.array([[0.1]])
C1 = np.array([[1.0]])
D1 = np.array([[0.0]])
# Feedback: proportional gain K=2
A2 = np.array([[0]])
B2 = np.array([[1]])
C2 = np.array([[2]])
D2 = np.array([[0]])
# Negative feedback (default)
Acl, Bcl, Ccl, Dcl = ss_feedback(A1, B1, C1, D1, A2, B2, C2, D2)
```

**See Also:** [ss_series](#ss_series), [ss_parallel](#ss_parallel), [place](#place)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.), Chapter 5
- [StackExchange: LTI interconnections](https://math.stackexchange.com/questions/2933407/formula-for-interconnection-of-ltis-in-state-space-form)
- [Wikipedia: Feedback](https://en.wikipedia.org/wiki/Feedback)

---

## Utilities

### pretty_polynomial

```python
pretty_polynomial(poly, symbol='q', decimals=2, tol=1e-10)
```

Format polynomial coefficients as a readable string.

Creates a nicely formatted string representation of a polynomial from its coefficient vector, useful for displaying transfer functions and characteristic equations.

**Parameters:**
- `poly` : ndarray, shape (n+1,) - Polynomial coefficients in descending order
- `symbol` : str, optional - Variable symbol to use (default is 'q' for z-transform). Common choices: 'q' or 'z' for discrete-time, 's' for continuous-time
- `decimals` : int, optional - Number of decimal places to display (default is 2)
- `tol` : float, optional - Tolerance for considering coefficients as zero (default is 1e-10)

**Returns:**
- `str_poly` : str - Formatted polynomial string

**Notes:**

- Coefficients near zero (within tol) are skipped
- Coefficient of 1.0 is omitted (prints "q^2" not "1.0q^2")
- Automatically adds '+' signs between positive terms

**Example:**
```python
# Discrete-time characteristic polynomial: z^2 - 1.5z + 0.5
poly = np.array([1, -1.5, 0.5])
print(pretty_polynomial(poly, symbol='z'))
# Output: z^2-1.5z^1+0.5z^0

# Transfer function numerator: 2s + 3
poly = np.array([2, 3])
print(pretty_polynomial(poly, symbol='s', decimals=1))
# Output: 2.0s^1+3.0s^0

# With very small coefficient (will be omitted)
poly = np.array([1, 1e-12, 0.5])
print(pretty_polynomial(poly, symbol='q'))
# Output: q^2+0.5q^0
```

**See Also:** [numpy.poly](https://numpy.org/doc/stable/reference/generated/numpy.poly.html), [c2d_characteristic_equation](#c2d_characteristic_equation), [place](#place)

**References:**
- Åström & Wittenmark (1997). Computer-Controlled Systems: Theory and Design (3rd ed.)

---

## Additional References

### Textbooks
1. Åström, K. J., & Wittenmark, B. (1997). *Computer-Controlled Systems: Theory and Design* (3rd ed.). Prentice Hall.
2. Franklin, G. F., Powell, J. D., & Workman, M. L. (1998). *Digital Control of Dynamic Systems* (3rd ed.). Addison-Wesley.

### Online Resources
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Python Control Systems Library](https://python-control.readthedocs.io/)

---

**License:** GNU General Public License (GNU GPLv3)

**Author Contact:** Rufus Fraanje - p.r.fraanje@hhs.nl
