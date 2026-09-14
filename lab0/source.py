import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


##EXPERIMENT 1
theta_i = np.array([0, 15, 30, 45, 60, 80], dtype=float)
trials = np.array([
    [0.0,  0.0,  0.0],
    [15.0, 15.5, 13.5],
    [30.0, 32.0, 31.5],
    [46.0, 46.0, 44.5],
    [59.5, 59.5, 58.5],
    [80.5, 79.5, 79.5],
])

theta_r = trials.mean(axis=1)
sem = trials.std(axis=1, ddof=1) / np.sqrt(trials.shape[1])

sigma_i = np.full_like(theta_i, 0.5)
sigma_r = np.maximum(sem, 0.5) 


residuals = theta_r - theta_i
chi2 = np.sum(residuals**2 / (sigma_r**2 + sigma_i**2))
dof = theta_i.size
reduced_chi2 = chi2 / dof

intersects = np.abs(residuals) <= np.maximum(sigma_i, sigma_r)
fraction = intersects.mean()

print(f"Chi-squared:         {chi2:.3f}")
print(f"Degrees of freedom:  {dof}")
print(f"Reduced chi-squared: {reduced_chi2:.3f}")
print(f"Error-bar intersections: {intersects.sum()}/{theta_i.size}")
print(f"Meets the 2/3 criterion: {fraction >= 2/3}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(
    theta_i, theta_r,
    xerr=sigma_i, yerr=sigma_r,
    fmt="o", capsize=4, markersize=5,
    label="Mean measured angles",
)
ax.plot([0, 85], [0, 85], "--", label=r"Prediction: $\theta_r=\theta_i$")

ax.set(
    xlabel=r"Incident angle $\theta_i$ (degrees)",
    ylabel=r"Reflected angle $\theta_r$ (degrees)",
    title="Law of reflection",
    xlim=(-3, 85),
    ylim=(-3, 85),
)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("experiment1-reflection.png", dpi=300)
plt.show()

##EXPERIMENT 2
angles = np.array([0, 10, 21.5, 30.5, 44.5, 56, 61, 76.5, 65.5, 48])
separations = np.array([0, 0.5, 1, 1.5, 2, 2.5, 2.7, 3, 2.9, 2.2])
separation_errors = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3])

t = 1.8817
t_error = 0.0028
angle_error = 0.5
accepted_n = 1.53

valid = separations > 0
angles = angles[valid]
y = separations[valid]
y_error = separation_errors[valid]

theta = np.radians(angles)
theta_error = np.radians(angle_error)

length = np.sqrt(y**2 + 4 * t**2)
n = np.sin(theta) * length / y

dn_dtheta = np.cos(theta) * length / y
dn_dt = 4 * t * np.sin(theta) / (y * length)
dn_dy = -4 * t**2 * np.sin(theta) / (y**2 * length)

n_error = np.sqrt(
    (dn_dtheta * theta_error)**2
    + (dn_dt * t_error)**2
    + (dn_dy * y_error)**2
)

mean_n = np.mean(n)
sd_n = np.std(n, ddof=1)
sem_n = sd_n / np.sqrt(len(n))

difference = abs(mean_n - accepted_n) / sem_n

lines = ["Angle (deg)    Separation (cm)    n        Uncertainty"]

for angle, separation, index, error in zip(angles, y, n, n_error):
    lines.append(
        f"{angle:8.1f}       {separation:8.1f}       "
        f"{index:.4f}    {error:.4f}"
    )

lines.extend([
    "",
    f"Mean refractive index: {mean_n:.4f}",
    f"Sample standard deviation: {sd_n:.4f}",
    f"Standard error: {sem_n:.4f}",
    f"Final result: {mean_n:.4f} +/- {sem_n:.4f}",
    f"Accepted value: {accepted_n:.2f}",
    f"Difference in standard errors: {difference:.2f}",
    f"Within two standard errors: {difference <= 2}",
    f"Within three standard errors: {difference <= 3}",
    "",
    "The standard error does not include shared systematic errors.",
])

results = "\n".join(lines)
print(results)

with open("experiment2-results.txt", "w", encoding="utf-8") as file:
    file.write(results)

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(
    angles, n,
    xerr=angle_error,
    yerr=n_error,
    fmt="o",
    color="black",
    capsize=4,
    label="Measured indices",
)

ax.axhline(mean_n, color="blue", label=f"Mean = {mean_n:.4f}")
ax.axhline(
    mean_n + sem_n,
    color="blue",
    linestyle="--",
    label="Mean ± standard error",
)
ax.axhline(mean_n - sem_n, color="blue", linestyle="--")
ax.axhline(accepted_n, color="orange", label="Accepted value = 1.53")

ax.set_xlabel("Incident angle (degrees)")
ax.set_ylabel("Refractive index")
ax.set_title("Experiment 2: Index of Refraction")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("experiment2-index.png", dpi=300)
plt.show()