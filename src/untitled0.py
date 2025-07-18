# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:56:38 2025

@author: oeham
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 461.5  # Gas constant for water vapor (J/kg·K)
gamma = 1.2  # Specific heat ratio for exhaust (LOX/LH2)
T_exhaust = 3000  # Exhaust temperature (K)
P_exhaust = 10000  # Nozzle exit pressure (Pa)
v_exhaust = 4400  # Exhaust velocity (m/s)
A_exit = 1.0  # Nozzle exit area (m^2)
m_dot = 500  # Mass flow rate (kg/s)
E_HMX = 5.7e6  # HMX energy release (J/kg)
m_HMX = 0.001  # HMX mass per detonation (kg)
freq = 100  # Detonations per second (Hz)
t_sim = 1.0  # Simulation time (s)
dt = 0.001  # Time step (s)

# Baseline thrust
F_baseline = m_dot * v_exhaust + (P_exhaust * A_exit)
print(f"Baseline Thrust: {F_baseline / 1000:.2f} kN")

# Simulation setup
t = np.arange(0, t_sim, dt)
P = np.ones_like(t) * P_exhaust  # Pressure array
F = np.ones_like(t) * F_baseline  # Thrust array
n_steps = len(t)

# Detonation model
for i in range(1, n_steps):
    # Check if detonation occurs (every 1/freq seconds)
    if i % int(1 / freq / dt) == 0:
        # Energy released by HMX detonation
        E_det = m_HMX * E_HMX
        # Assume energy increases exhaust temperature in control volume
        dT = E_det / (m_dot * dt * 718)  # Approx. specific heat of water vapor (J/kg·K)
        T_new = T_exhaust + dT
        # New pressure from ideal gas law (P ~ T for constant volume)
        P_new = P_exhaust * (T_new / T_exhaust)
        P[i] = P_new
        # New exhaust velocity (approximate, based on energy addition)
        v_new = np.sqrt(v_exhaust**2 + 2 * E_det / (m_dot * dt))
        # New thrust
        F[i] = m_dot * v_new + (P_new * A_exit)
    else:
        P[i] = P_exhaust
        F[i] = F_baseline

# Calculate average thrust
F_avg = np.mean(F)
thrust_increase = (F_avg - F_baseline) / F_baseline * 100
print(f"Average Thrust: {F_avg / 1000:.2f} kN")
print(f"Thrust Increase: {thrust_increase:.2f}%")

# Plot pressure over time
plt.figure(figsize=(10, 6))
plt.plot(t, P / 1000, label="Nozzle Exit Pressure")
plt.axhline(P_exhaust / 1000, color='r', linestyle='--', label="Baseline Pressure")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (kPa)")
plt.title("Nozzle Pressure with HMX Detonations")
plt.legend()
plt.grid(True)
plt.savefig("nozzle_pressure.png")