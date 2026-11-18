"""Numerics for the main text of "Quantum Measurement Incompatibility Multiplies Sample Complexity: Minimax Bounds for Model Selection".

* ``operators``  anticommuting Pauli families and exact checks of the joint POVM
* ``sdp``        joint-measurability thresholds by semidefinite programming
* ``selection``  Monte Carlo selection experiments on the hard instances of H_{m,G}
* ``pairs``      Monte Carlo estimation experiments for weakly incompatible qubit pairs
* ``kappa``      incompatibility number: thresholds + covering LP (Algorithm 1), Lambda, 5-cycle, random families
* ``checks``     Born-rule check of the binomial model and typical (non-adversarial) instances
* ``style``      matplotlib style of the paper figures

The appendix (noisy circuit simulation of the randomized-basis implementation) lives in
the separate ``jointpovm`` package.
"""
