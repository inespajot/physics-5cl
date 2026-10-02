import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).parent

angle = np.array([49, 51, 53, 55, 57, 59])
intensity = np.array([1343, 1390, 1408, 1420, 1342, 1247])
error = np.array([3, 3, 3, 3, 3, 3])

a, b, c = np.polyfit(angle, intensity, 2, w=1/error)

theta_B = -b / (2*a)
n = np.tan(np.radians(theta_B))

fitted = a*angle**2 + b*angle + c
residuals = intensity - fitted

chi_squared = np.sum((residuals/error)**2)
reduced_chi_squared = chi_squared / (len(angle) - 3)

x = np.linspace(49, 59, 300)
y = a*x**2 + b*x + c

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

ax1.errorbar(
    angle, intensity,
    xerr=1,
    yerr=error,
    fmt="o",
    capsize=3,
    label="Data"
)
ax1.plot(x, y, label="Quadratic fit")
ax1.axvline(
    theta_B,
    linestyle="--",
    label=fr"$\theta_B={theta_B:.1f}^\circ$"
)
ax1.set_ylabel("Corrected intensity (lux)")
ax1.legend()
ax1.grid()

ax2.errorbar(
    angle, residuals,
    xerr=1,
    yerr=error,
    fmt="o",
    capsize=3
)
ax2.axhline(0, color="black", linestyle="--")
ax2.set_xlabel("Incident angle (degrees)")
ax2.set_ylabel("Residuals (lux)")
ax2.grid()

plt.tight_layout()
plt.savefig(folder / "brewster_fit.png", dpi=300, bbox_inches="tight")
plt.show()

print("Brewster angle =", theta_B, "degrees")
print("Index of refraction =", n)
print("Chi-squared =", chi_squared)
print("Reduced chi-squared =", reduced_chi_squared)