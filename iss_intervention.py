import numpy as np

def detect_collapse(m, r, p, a, d):
    """
    Simple collapse detector.
    m = model stability
    r = resilience
    p = pressure
    a = amplification
    d = decay

    Returns:
        0 if collapse detected
        None otherwise
    """
    stability = m * (1 - p) - (d * (p * a))
    if stability < 0:
        return 0
    return None


def simulate_intervention(pressure, intervention_step, total_steps=60):
    """
    Simulates stability over time under pressure with optional intervention.

    pressure: external load on the system
    intervention_step: timestep at which intervention begins
    total_steps: length of simulation
    """
    S = np.zeros(total_steps)
    S[0] = 1.0  # initial stability

    for i in range(1, total_steps):
        decay = pressure * 0.4
        recovery = 0.0

        if intervention_step is not None and i >= intervention_step:
            recovery = 0.8

        S[i] = S[i-1] - decay + recovery

    return S


def compare_scenarios():
    """
    Returns four intervention scenarios:
    - baseline: no intervention
    - early: intervention at step 3
    - optimal: intervention at step 6
    - late: intervention at step 12
    """
    baseline = simulate_intervention(pressure=0.6, intervention_step=None)
    early = simulate_intervention(pressure=0.6, intervention_step=3)
    optimal = simulate_intervention(pressure=0.6, intervention_step=6)
    late = simulate_intervention(pressure=0.6, intervention_step=12)

    return {
        "baseline": baseline,
        "early": early,
        "optimal": optimal,
        "late": late,
    }
