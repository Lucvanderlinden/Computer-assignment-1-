import numpy as np
import matplotlib.pyplot as plt
from lj_dimer import velocity_verlet_integrator, normalized_lj_potential


def total_energy(p, x):
    """Compute total energy: kinetic + potential"""
    kinetic = np.array([i**2 / 2 for i in p])  # (1/2)p²
    potential = np.array([normalized_lj_potential(j) for j in x])
    return kinetic + potential


# Part 1a: Energy Conservation
print("=" * 60)
print("Part 1a: Energy Conservation")
print("=" * 60)

# Simulate with default dt and plot energy
print("\n1. Single trajectory energy conservation (dt=2e-4)")
positions, momenta, time = velocity_verlet_integrator(
    initial_position=1.3,
    force_function=lambda rho: 12.0 * (rho ** (-12) - rho ** (-6)) / rho,
    dt=2e-4
)

energy = total_energy(momenta, positions)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time, energy, color='blue', linewidth=0.5)
ax.set(
    xlabel='time',
    ylabel='total energy E(t)',
    title=f'Energy Conservation (dt=2e-4)'
)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig('part_1a_single_trajectory_energy.png')
plt.show()

# Analyze energy drift
energy_drift = abs(energy[-1] - energy[0])
print(f"Energy drift for dt=2e-4: {energy_drift:.2e}")
print(f"Initial energy: {energy[0]:.6f}")
print(f"Final energy: {energy[-1]:.6f}")
print(f"Mean energy: {np.mean(energy):.6f}")
print(f"Energy std dev: {np.std(energy):.2e}")


# Study energy conservation vs dt
print("\n2. Energy conservation vs time-step (dt)")
dt_values = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
energy_drifts = []
mean_energies = []

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for dt in dt_values:
    print(f"   Simulating with dt = {dt:.0e}...", end=" ", flush=True)
    pos, mom, t = velocity_verlet_integrator(
        initial_position=1.3,
        force_function=lambda rho: 12.0 * (rho ** (-12) - rho ** (-6)) / rho,
        dt=dt
    )
    E = total_energy(mom, pos)
    drift = abs(E[-1] - E[0])
    energy_drifts.append(drift)
    mean_energies.append(np.mean(E))
    
    # Plot a few trajectories
    axs[0].plot(t, E, label=f'dt={dt:.0e}', linewidth=0.8, alpha=0.7)
    print("done")

# Plot drift vs dt
axs[1].loglog(dt_values, energy_drifts, 'o-', color='red', markersize=6, label='Energy drift')
axs[1].set(xlabel='time-step (dt)', ylabel='energy drift |E_final - E_initial|', 
           title='Energy Conservation Error vs Time-step')
axs[1].grid(True, alpha=0.3, which='both')
axs[0].set(xlabel='time', ylabel='total energy E(t)', title='Energy vs Time (Different dt)')
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

fig.tight_layout()
plt.savefig('part_1a_energy_vs_dt.png')
plt.show()

print("\n" + "=" * 60)
print("Energy Conservation Analysis Summary")
print("=" * 60)
print(f"{'dt':<12} {'E_drift':<15} {'Mean E':<12}")
print("-" * 60)
for dt, drift, mean_E in zip(dt_values, energy_drifts, mean_energies):
    print(f"{dt:<12.0e} {drift:<15.2e} {mean_E:<12.6f}")
print("=" * 60)
print("Note: Smaller dt → better energy conservation")
print("Expected scaling: drift ~ O(dt²) for Velocity Verlet")
