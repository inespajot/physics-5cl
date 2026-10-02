import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

folder = Path(__file__).parent

angle = np.arange(0, 181, 15)

intensity = np.array([
    4111, 3925, 3318, 2288, 1114, 44, 0,
    0, 278, 1448, 2665, 3727, 4255
])

error = np.array([
    71, 72, 71, 71, 72, 71, 71,
    71, 71, 71, 71, 73, 71
])

def malus(theta, I0, theta0, Ibg):
    return I0 * np.cos(np.radians(theta - theta0))**2 + Ibg

def analyze(mask, filename):
    fit, _ = curve_fit(
        malus,
        angle[mask],
        intensity[mask],
        sigma=error[mask],
        absolute_sigma=True,
        p0=[4200, 0, 0]
    )

    residuals = intensity - malus(angle, *fit)
    reduced_chi2 = np.sum(
        (residuals[mask] / error[mask])**2
    ) / (np.sum(mask) - 3)

    x = np.linspace(0, 180, 500)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True
    )
    
    top.errorbar(
    angle[mask],
    intensity[mask],
    xerr=1,
    yerr=error[mask],
    fmt="o",
    capsize=3,
    label="Data"
    )

    top.plot(x, malus(x, *fit), label="Fit")
    top.set_ylabel("Intensity (lux)")
    top.legend()
    top.grid()

    bottom.errorbar(
    angle[mask],
    residuals[mask],
    xerr=1,
    yerr=error[mask],
    fmt="o",
    capsize=3
)   
    bottom.axhline(0, color="black", linestyle="--")
    bottom.set_xlabel("Angle (degrees)")
    bottom.set_ylabel("Residuals (lux)")
    bottom.grid()

    plt.tight_layout()
    plt.savefig(folder / filename, dpi=300)
    plt.show()

    print(filename)
    print("I0 =", fit[0], "lux")
    print("theta0 =", fit[1], "degrees")
    print("Ibg =", fit[2], "lux")
    print("Minimum angle =", (fit[1] + 90) % 180, "degrees")
    print("Reduced chi-squared =", reduced_chi2)


analyze(
    np.ones(len(angle), dtype=bool),
    "malus_all_data.png"
)


analyze(
    ~np.isin(angle, [90, 105]),
    "malus_without_center.png"
)