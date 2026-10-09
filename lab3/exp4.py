import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Diffraction orders
m = np.array([1, 2, 3, 4, 5, 6])

# Full distances from -m to +m, in cm
distances = {
    "26-gauge": np.array([0.27, 0.63, 0.73, 0.97, 1.19, 1.45]),
    "39-gauge": np.array([0.87, 1.72, 2.69, 3.47, 4.39, 5.28])
}

# Measurement uncertainty
sigma_D = 0.014       # cm
sigma_y = sigma_D / 2 # cm

# Green laser wavelength
wavelength = 532e-9   # m

# Set to zero if no wavelength uncertainty was provided
sigma_wavelength = 0  # m

# Wire and screen positions
wire_position_cm = 6.2
screen_position_cm = 95.0
sigma_position_cm = 0.05

L_cm = screen_position_cm - wire_position_cm

sigma_L_cm = np.sqrt(
    sigma_position_cm**2
    + sigma_position_cm**2
)

L = L_cm / 100
sigma_L = sigma_L_cm / 100

# Accepted wire diameters in mm
accepted_diameters = {
    "26-gauge": 0.405,
    "39-gauge": 0.0897
}

# One unit in the last reported digit
accepted_uncertainties = {
    "26-gauge": 0.001,
    "39-gauge": 0.0001
}


# Linear model through the origin
def model(m, slope):
    return slope * m


fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7),
    sharex="col",
    gridspec_kw={"height_ratios": [3, 1]}
)

print(f"L = {L_cm:.1f} +/- {sigma_L_cm:.2f} cm")
print(f"Wavelength = {wavelength * 1e9:.0f} nm")
print()

for column, (wire, full_distance) in enumerate(distances.items()):

    # Convert full distance to one-sided minimum position
    y = full_distance / 2
    y_uncertainties = np.full(len(y), sigma_y)

    # Fit y_m = slope * m
    parameters, covariance = curve_fit(
        model,
        m,
        y,
        sigma=y_uncertainties,
        absolute_sigma=True
    )

    slope = parameters[0]
    original_slope_uncertainty = np.sqrt(covariance[0, 0])

    # Fitted values and residuals
    fitted_y = model(m, slope)
    residuals = y - fitted_y

    # Chi-squared
    chi_squared = np.sum(
        (residuals / y_uncertainties) ** 2
    )

    degrees_of_freedom = len(y) - 1
    reduced_chi_squared = chi_squared / degrees_of_freedom

    # Increase slope uncertainty when reduced chi-squared exceeds 1
    uncertainty_scale = np.sqrt(
        max(1, reduced_chi_squared)
    )

    slope_uncertainty = (
        original_slope_uncertainty
        * uncertainty_scale
    )

    # Convert slope from cm to m
    slope_m = slope / 100
    slope_uncertainty_m = slope_uncertainty / 100

    # Calculate wire diameter
    diameter_m = wavelength * L / slope_m
    diameter_mm = diameter_m * 1000

    # Propagate wavelength, L, and slope uncertainties
    relative_uncertainty = np.sqrt(
        (sigma_wavelength / wavelength) ** 2
        + (sigma_L / L) ** 2
        + (slope_uncertainty_m / slope_m) ** 2
    )

    diameter_uncertainty_mm = (
        diameter_mm * relative_uncertainty
    )

    # Accepted diameter and uncertainty
    accepted = accepted_diameters[wire]
    accepted_uncertainty = accepted_uncertainties[wire]

    # Agreement test
    difference = abs(diameter_mm - accepted)

    agreement_limit = 2 * np.sqrt(
        diameter_uncertainty_mm**2
        + accepted_uncertainty**2
    )

    agrees = difference <= agreement_limit

    # Print results
    print(wire)

    print(
        f"Slope = {slope:.4f} +/- "
        f"{slope_uncertainty:.4f} cm"
    )

    print(
        f"Measured diameter = {diameter_mm:.4f} +/- "
        f"{diameter_uncertainty_mm:.4f} mm"
    )

    print(
        f"Accepted diameter = {accepted:.4f} +/- "
        f"{accepted_uncertainty:.4f} mm"
    )

    print(f"Chi-squared = {chi_squared:.2f}")

    print(
        f"Reduced chi-squared = "
        f"{reduced_chi_squared:.2f}"
    )

    print(f"|z1 - z2| = {difference:.4f} mm")

    print(
        f"Agreement limit = "
        f"{agreement_limit:.4f} mm"
    )

    print(
        f"Agreement = "
        f"{'Yes' if agrees else 'No'}"
    )

    print()

    # Main fit graph
    main_ax = axes[0, column]

    main_ax.errorbar(
        m,
        y,
        yerr=y_uncertainties,
        fmt="o",
        capsize=4,
        label="Data"
    )

    m_fit = np.linspace(0, 6.3, 100)

    main_ax.plot(
        m_fit,
        model(m_fit, slope),
        label="Fit"
    )

    main_ax.set_title(wire)
    main_ax.set_ylabel(r"$y_m$ (cm)")
    main_ax.grid(alpha=0.3)
    main_ax.legend()

    # Residual graph
    residual_ax = axes[1, column]

    residual_ax.errorbar(
        m,
        residuals,
        yerr=y_uncertainties,
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
plt.savefig(
    "experiment4_fits.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()