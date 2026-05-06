import numpy as np


def normalized_lj_potential(rho):
    rho_inverse_6 = rho ** (-6)
    potential = rho_inverse_6**2 - 2 * rho_inverse_6
    return potential


def normalized_lj_force(rho):
    rho_inverse_6 = rho ** (-6)
    force = 12.0 * (rho_inverse_6**2 - rho_inverse_6) / rho
    return force


def velocity_verlet_integrator(
    initial_position, force_function, dt=2e-4, max_steps=5_000_000
):
    """
    Integrate one closed orbit using the velocity-Verlet algorithm.

    Starting from (q(0), p(0)) = (q0, 0), the update rule is:

        p_{n+1/2} = p_n       + (dt/2) * F(q_n)
        q_{n+1}   = q_n       + dt     * p_{n+1/2}
        p_{n+1}   = p_{n+1/2} + (dt/2) * F(q_{n+1})

    Integration stops when p changes sign for the second time
    (one full oscillation). Returns None if no orbit is found.
    """

    q, p, t = float(initial_position), 0.0, 0.0
    qs, ps, ts = [q], [p], [t]
    p_prev = 0.0
    zero_crossings = 0

    for _ in range(max_steps):
        # velocity verlet step
        p_half = p + 0.5 * dt * force_function(q)
        q = q + dt * p_half
        p = p_half + 0.5 * dt * force_function(q)
        t += dt

        # accumulate values
        qs.append(q)
        ps.append(p)
        ts.append(t)

        # check momentum sign-change
        if len(qs) > 2 and p_prev * p < 0.0:
            zero_crossings += 1
            # break after complete orbit
            if zero_crossings == 2:
                break
        p_prev = p
    else:
        return None

    return np.array(qs), np.array(ps), np.array(ts)
