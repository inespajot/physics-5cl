import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m = np.array([1, 2, 3])

D = {
    "2 pt": np.array([0.81, 1.51, 2.24]),
    "4 pt": np.array([0.46, 0.89, 1.35]),
    "8 pt": np.array([0.26, 0.47, 0.84])
}

sigma_D = 0.014       # cm
sigma_y = sigma_D / 2 # cm
wavelength = 635e-9   # m
L = (95.0 - 5.9) / 100  # m

# mm
nominal_widths = {
    "2 pt": 2 * 0.04393,
    "4 pt": 4 * 0.04393,
    "8 pt": 8 * 0.04393
}

def model(m, slope):
    return slope * m

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 7),
    sharex="col",
    gridspec_kw={"height_ratios": [3, 1]}
)

for column, (slit, full_distance) in enumerate(D.items()):

    y = full_distance / 2

    uncertainties = np.full(len(y), sigma_y)

    parameters, covariance = curve_fit(
        model,
        m,
        y,
        sigma=uncertainties,
        absolute_sigma=True
    )

    slope = parameters[0]
    slope_uncertainty = np.sqrt(covariance[0, 0])

    fitted_y = model(m, slope)
    residuals = y - fitted_y

    chi_squared = np.sum((residuals / uncertainties) ** 2)
    degrees_of_freedom = len(y) - 1
    reduced_chi_squared = chi_squared / degrees_of_freedom

    slope_m = slope / 100
    slope_uncertainty_m = slope_uncertainty / 100

    slit_width_m = wavelength * L / slope_m
    slit_width_mm = slit_width_m * 1000

    width_uncertainty_mm = (
        slit_width_mm * slope_uncertainty_m / slope_m
    )

    nominal = nominal_widths[slit]

    percent_difference = (
        abs(slit_width_mm - nominal) / nominal * 100
    )

    print(slit)
    print(f"Slope = {slope:.4f} +/- {slope_uncertainty:.4f} cm")
    print(
        f"Measured width = "
        f"{slit_width_mm:.3f} +/- {width_uncertainty_mm:.3f} mm"
    )
    print(f"Nominal width = {nominal:.3f} mm")
    print(f"Percent difference = {percent_difference:.1f}%")
    print(f"Reduced chi-squared = {reduced_chi_squared:.2f}")
    print()

    # Main graph
    main_ax = axes[0, column]

    main_ax.errorbar(
        m,
        y,
        yerr=uncertainties,
        fmt="o",
        capsize=4,
        label="Data"
    )

    m_fit = np.linspace(0, 3.2, 100)

    main_ax.plot(
        m_fit,
        model(m_fit, slope),
        label="Fit"
    )

    main_ax.set_title(slit)
    main_ax.set_ylabel(r"$y_m$ (cm)")
    main_ax.grid(alpha=0.3)
    main_ax.legend()

    # Residual graph
    residual_ax = axes[1, column]

    residual_ax.errorbar(
        m,
        residuals,
        yerr=uncertainties,
        fmt="o",
        capsize=4
    )

    residual_ax.axhline(
        0,
        color="black",
        linestyle="--"
    )

    residual_ax.set_xlabel(r"Order $m$")
    residual_ax.set_ylabel("Residual (cm)")
    residual_ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("experiment3_fits.png", dpi=300)
plt.show()