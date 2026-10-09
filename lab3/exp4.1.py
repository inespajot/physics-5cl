import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Diffraction orders
m = np.array([1, 2, 3, 4, 5, 6])

# Full distances
distances = {
    "26-gauge": np.array([0.27, 0.63, 0.73, 0.97, 1.19, 1.45]),
    "39-gauge": np.array([0.87, 1.72, 2.69, 3.47, 4.39, 5.28]),
}

# Measurement uncertainty
sigma_D = 0.014  # cm
sigma_y = sigma_D / 2  # cm

# Green laser wavelength
wavelength = 532e-9  # m
sigma_wavelength = 0  # m

# Wire and screen positions
wire_position_cm = 6.2
screen_position_cm = 95.0
sigma_position_cm = 0.05

L_cm = screen_position_cm - wire_position_cm
sigma_L_cm = np.sqrt(2 * sigma_position_cm**2)
L = L_cm / 100
sigma_L = sigma_L_cm / 100

# Accepted wire diameters in mm
accepted_diameters = {
    "26-gauge": 0.405,
    "39-gauge": 0.0897,
}

accepted_uncertainties = {
    "26-gauge": 0.001,
    "39-gauge": 0.0001,
}


def model(order, slope):
    """Linear diffraction model through the origin."""
    return slope * order


fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7),
    sharex="col",
    gridspec_kw={"height_ratios": [3, 1]},
)

print(f"L = {L_cm:.1f} +/- {sigma_L_cm:.2f} cm")
print(f"Wavelength = {wavelength * 1e9:.0f} nm")
print()

for column, (wire, full_distance) in enumerate(distances.items()):
    y_all = full_distance / 2

    # Exclude m = 2 for the 26-gauge wire from the fit.
    if wire == "26-gauge":
        fit_mask = m != 2
    else:
        fit_mask = np.ones(len(m), dtype=bool)

    m_used = m[fit_mask]
    y_used = y_all[fit_mask]
    uncertainties_used = np.full(len(y_used), sigma_y)

    parameters, covariance = curve_fit(
        model,
        m_used,
        y_used,
        sigma=uncertainties_used,
        absolute_sigma=True,
    )

    slope = parameters[0]
    original_slope_uncertainty = np.sqrt(covariance[0, 0])

    fitted_y = model(m_used, slope)
    residuals = y_used - fitted_y

    chi_squared = np.sum((residuals / uncertainties_used) ** 2)
    degrees_of_freedom = len(y_used) - 1
    reduced_chi_squared = chi_squared / degrees_of_freedom

    uncertainty_scale = np.sqrt(max(1, reduced_chi_squared))
    slope_uncertainty = original_slope_uncertainty * uncertainty_scale

    slope_m = slope / 100
    slope_uncertainty_m = slope_uncertainty / 100

    diameter_m = wavelength * L / slope_m
    diameter_mm = diameter_m * 1000

    relative_uncertainty = np.sqrt(
        (sigma_wavelength / wavelength) ** 2
        + (sigma_L / L) ** 2
        + (slope_uncertainty_m / slope_m) ** 2
    )
    diameter_uncertainty_mm = diameter_mm * relative_uncertainty

    # Agreement test
    accepted = accepted_diameters[wire]
    accepted_uncertainty = accepted_uncertainties[wire]
    difference = abs(diameter_mm - accepted)
    agreement_limit = 2 * np.sqrt(
        diameter_uncertainty_mm**2 + accepted_uncertainty**2
    )
    agrees = difference <= agreement_limit

    print(wire)
    print(f"Slope = {slope:.4f} +/- {slope_uncertainty:.4f} cm")
    print(
        f"Measured diameter = {diameter_mm:.4f} +/- "
        f"{diameter_uncertainty_mm:.4f} mm"
    )
    print(
        f"Accepted diameter = {accepted:.4f} +/- "
        f"{accepted_uncertainty:.4f} mm"
    )
    print(f"Chi-squared = {chi_squared:.2f}")
    print(f"Reduced chi-squared = {reduced_chi_squared:.2f}")
    print(f"|z1 - z2| = {difference:.4f} mm")
    print(f"Agreement limit = {agreement_limit:.4f} mm")
    print(f"Agreement = {'Yes' if agrees else 'No'}")
    print()

    # Main fit graph
    main_ax = axes[0, column]
    main_ax.errorbar(
        m_used,
        y_used,
        yerr=uncertainties_used,
        fmt="o",
        capsize=4,
        label="Included data",
    )

    if np.any(~fit_mask):
        main_ax.plot(
            m[~fit_mask],
            y_all[~fit_mask],
            "rx",
            markersize=8,
            label="Excluded point",
        )

    m_fit = np.linspace(0, 6.3, 100)
    main_ax.plot(m_fit, model(m_fit, slope), label="Fit")
    main_ax.set_title(wire)
    main_ax.set_ylabel(r"$y_m$ (cm)")
    main_ax.grid(alpha=0.3)
    main_ax.legend()

    # Residual graph
    residual_ax = axes[1, column]
    residual_ax.errorbar(
        m_used,
        residuals,
        yerr=uncertainties_used,
        fmt="o",
        capsize=4,
    )
    residual_ax.axhline(0, color="black", linestyle="--")
    residual_ax.set_xlabel(r"Order $m$")
    residual_ax.set_ylabel("Residual (cm)")
    residual_ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("experiment4_fits2.png", dpi=300, bbox_inches="tight")
plt.show()