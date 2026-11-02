"""Render the deterministic PI/spectroscopy review results with Matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
data = json.loads((root/"pi_spectroscopy_results.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3), constrained_layout=True)

ax = axes[0, 0]
rows = data["suzuki_chin"]["rows"]
p = np.array([r["beads"] for r in rows])
ax.loglog(p, [r["primitive_absolute_error"] for r in rows], "o-", color="#4477AA", label="Primitive (order 1.994)")
ax.loglog(p, [r["sc_absolute_error"] for r in rows], "s-", color="#CC6677", label="SC (order 3.989)")
ax.set(title="A. Nonharmonic thermal potential", xlabel="Number of beads, P", ylabel="Absolute error versus Schrödinger reference")
ax.legend(frameon=False)

ax = axes[0, 1]
rows = data["pigs"]["tau_sweep"]
tau = np.array([r["tau"] for r in rows])
error = np.array([r["absolute_error"] for r in rows])
ax.loglog(tau, error, "o-", color="#228833", label="PIGS at projection T=8")
ax.loglog(tau, error[-1]*(tau/tau[-1])**2, "--", color="#666666", label="Second-order guide")
ax.set(title="B. Finite imaginary-time discretization", xlabel="Imaginary-time interval, tau", ylabel="Error versus exact finite projection")
ax.legend(frameon=False)

ax = axes[1, 0]
rows = data["pigs"]["projection_sweep"]
duration = [r["projection_time"] for r in rows]
ax.semilogy(duration, [abs(r["projection_error"]) for r in rows], "o-", color="#AA4499", label="Finite projection error")
ax.semilogy(duration, [abs(r["finite_tau_error"]) for r in rows], "s-", color="#DD9900", label="Finite tau error (tau=0.03125)")
ax.set(title="C. Two distinct PIGS error sources", xlabel="Total projection duration, T", ylabel="Absolute potential error")
ax.legend(frameon=False)

ax = axes[1, 1]
rows = data["pacmd"]["rows"]
dt = np.array([r["dt"] for r in rows])
error = np.array([r["max_centroid_error"] for r in rows])
ax.loglog(dt, error, "o-", color="#44AA99", label="Coupled 6D PA-CMD centroid")
ax.loglog(dt, error[-1]*(dt/dt[-1])**2, "--", color="#666666", label="Second-order guide")
ax.set(title="D. Harmonic centroid dynamics", xlabel="Real-time step, dt", ylabel="Maximum error at t=5")
ax.legend(frameon=False)

for ax in axes.flat:
    ax.grid(True, which="both", alpha=.18)
fig.suptitle("PR #63 independent numerical review\nMethod-level checks; no molecular-spectrum or sampling-convergence claim", fontsize=14)
fig.savefig(root/"pi_spectroscopy_convergence.png", dpi=180)
fig.savefig(root/"pi_spectroscopy_convergence.pdf")
print("pi_spectroscopy_convergence.png and .pdf written")
