import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont


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

##EXPERIMENT 3

do = np.array([29.40, 33.40, 38.40, 27.40, 43.40])
di = np.array([40.95, 33.05, 28.20, 47.00, 25.45])
hi = np.array([3.70, 2.65, 1.95, 4.65, 1.45])

do_error = np.full(5, 0.05)
di_error = np.array([0.45, 0.75, 0.50, 1.40, 0.15])
hi_error = np.array([0.10, 0.05, 0.05, 0.15, 0.05])

object_readings = np.array([2.230, 2.190, 2.240])
ho = object_readings.mean()
ho_error = object_readings.std(ddof=1) / np.sqrt(3)

x = 1 / do
y = 1 / di
x_error = do_error / do**2
y_error = di_error / di**2

covariance = np.cov(x, y, ddof=1)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

effective_error = np.hypot(x_error, y_error)
weights = 1 / effective_error**2

b = np.sum(weights * (x + y)) / np.sum(weights)
b_error = 1 / np.sqrt(np.sum(weights))

f = 1 / b
f_error = b_error / b**2

residuals = y - (-x + b)
chi2 = np.sum((residuals / effective_error)**2)
dof = len(x) - 1
reduced_chi2 = chi2 / dof

measured_M = -hi / ho
predicted_M = -di / do

measured_M_error = np.hypot(
    hi_error / ho,
    hi * ho_error / ho**2,
)
predicted_M_error = np.hypot(
    di_error / do,
    di * do_error / do**2,
)

absolute_difference = np.abs(measured_M - predicted_M)
twice_combined_error = 2 * np.hypot(
    measured_M_error, predicted_M_error
)
agreement = absolute_difference < twice_combined_error

print(f"Object dimension: {ho:.4f} +/- {ho_error:.4f} cm")
print(f"Sample covariance: {covariance:.6e} cm^-2")
print(f"Correlation coefficient: {correlation:.6f}")
print("Effective uncertainties (cm^-1):", effective_error)
print(f"Intercept: {b:.8f} +/- {b_error:.8f} cm^-1")
print(f"Focal length: {f:.3f} +/- {f_error:.3f} cm")
print(f"Chi-squared: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"Reduced chi-squared: {reduced_chi2:.2f}")

print(
    "\ndo (cm)   Measured M         Predicted M        "
    "Abs. difference   2 x combined uncertainty   Agreement"
)
for i in range(len(do)):
    print(
        f"{do[i]:6.2f}    "
        f"{measured_M[i]:.3f} +/- {measured_M_error[i]:.3f}    "
        f"{predicted_M[i]:.3f} +/- {predicted_M_error[i]:.3f}    "
        f"{absolute_difference[i]:.3f}            "
        f"{twice_combined_error[i]:.3f}                      "
        f"{'Yes' if agreement[i] else 'No'}"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].errorbar(
    x, y, xerr=x_error, yerr=y_error,
    fmt="o", capsize=4, label="Measurements",
)
x_line = np.linspace(x.min() - 0.001, x.max() + 0.001, 100)
axes[0].plot(x_line, -x_line + b, label="Fixed-slope fit")
axes[0].set_xlabel(r"$1/d_o$ (cm$^{-1}$)")
axes[0].set_ylabel(r"$1/d_i$ (cm$^{-1}$)")
axes[0].set_title("Thin-lens equation")

axes[1].errorbar(
    predicted_M, measured_M,
    xerr=predicted_M_error,
    yerr=measured_M_error,
    fmt="o", capsize=4, label="Measurements",
)
lower = min(
    (predicted_M - predicted_M_error).min(),
    (measured_M - measured_M_error).min(),
) - 0.1
upper = max(
    (predicted_M + predicted_M_error).max(),
    (measured_M + measured_M_error).max(),
) + 0.1

axes[1].plot(
    [lower, upper], [lower, upper], "--", label="Equality"
)
axes[1].set_xlabel("Predicted magnification")
axes[1].set_ylabel("Measured magnification")
axes[1].set_title("Magnification comparison")

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig("experiment3-analysis.png", dpi=300)
plt.show()

##EXPERIMENT 4
root = Path(__file__).resolve().parent
incident = np.arange(0, 81, 10.)
transmitted = np.array([[0,0],[6.5,7],[13.5,14],[20,19.5],[26,26.5],
                        [31,32],[36.5,36],[40,39],[41.5,41.5]])
a = np.deg2rad(incident)
r = np.deg2rad(transmitted.mean(axis=1))
sigma_angle = np.deg2rad(0.5)
x, y = np.sin(r), np.sin(a)
sx, sy = np.cos(r)*sigma_angle, np.cos(a)*sigma_angle
m = 1.5
for iteration in range(1000):
    w = 1/(sy**2 + m**2*sx**2)
    updated = np.sum(w*x*y)/np.sum(w*x*x)
    if abs(updated-m) < 1e-13:
        m = updated
        break
    m = updated
w = 1/(sy**2 + m**2*sx**2)
sigma_m = 1/np.sqrt(np.sum(w*x*x))
residual = y-m*x
chi2 = np.sum(w*residual**2)
dof = len(x)-1
# Survival probability for chi-square with eight degrees of freedom.
p = math.exp(-chi2/2)*sum((chi2/2)**k/math.factorial(k) for k in range(4))
n4, sn4 = 1.0003*m, 1.0003*sigma_m

thickness = np.array([1.876,1.885,1.884])
t = thickness.mean()
angles2 = np.deg2rad([10,21.5,30.5,44.5,56,61,76.5,65.5,48])
separations = np.array([.5,1,1.5,2,2.5,2.7,3,2.9,2.2])
indices2 = np.sin(angles2)*np.sqrt(separations**2+4*t*t)/separations
n2 = indices2.mean()
s2 = indices2.std(ddof=1)
sn2 = s2/np.sqrt(len(indices2))
critical = np.deg2rad(np.mean([43,43,43.5]))
nc = 1.0003/np.sin(critical)
snc = nc/np.tan(critical)*np.deg2rad(1)
summary = '\n'.join([
    f'slope = {m:.9f} +/- {sigma_m:.9f}',
    f'n4 = {n4:.9f} +/- {sn4:.9f}',
    f'chi2 = {chi2:.9f}; dof = {dof}; reduced chi2 = {chi2/dof:.9f}; p = {p:.9f}',
    f'Part 2 thickness mean (cm) = {t:.9f}',
    f'Part 2 individual indices = {indices2}',
    f'n2 = {n2:.9f}; sample SD = {s2:.9f}; SEM = {sn2:.9f}',
    f'critical index = {nc:.9f} +/- {snc:.9f}',
    f'Part 2 vs Part 4 normalized difference = {abs(n2-n4)/np.hypot(sn2,sn4):.9f}',
    f'Critical vs Part 4 normalized difference = {abs(nc-n4)/np.hypot(snc,sn4):.9f}',
])
print(summary)

im = Image.new('RGB', (1600,1500), 'white')
draw = ImageDraw.Draw(im)
font_dir = Path('C:/Windows/Fonts')
def font(size):
    try: return ImageFont.truetype(str(font_dir/'arial.ttf'),size)
    except OSError: return ImageFont.load_default(size=size)
f, title = font(29), font(39)
ink, blue, gray = '#202B38', '#176B9B', '#DCE2E8'
draw.text((160,35),'Snell\'s law: hemisphere measurements',fill=ink,font=title)
def panel(top,bottom,ymin,ymax,ylabel):
    left,right=170,1510
    def xy(xv,yv): return (left+(xv+.02)/.74*(right-left), bottom-(yv-ymin)/(ymax-ymin)*(bottom-top))
    for xv in np.arange(0,.71,.1):
        px,_=xy(xv,ymin)
        draw.line((px,top,px,bottom),fill=gray,width=2)
        draw.text((px-20,bottom+15),f'{xv:.1f}',fill=ink,font=f)
    for yv in np.linspace(ymin,ymax,6):
        _,py=xy(0,yv)
        draw.line((left,py,right,py),fill=gray,width=2)
        draw.text((18,py-18),f'{yv:.2f}',fill=ink,font=f)
    draw.rectangle((left,top,right,bottom),outline=ink,width=3)
    draw.text((left,top-47),ylabel,fill=ink,font=f)
    return xy
xy=panel(165,820,-.025,1.10,'sin(theta_in)')
draw.line((*xy(0,0),*xy(.7,m*.7)),fill=blue,width=5)
for xx,yy,ex,ey in zip(x,y,sx,sy):
    px,py=xy(xx,yy)
    draw.line((*xy(xx-ex,yy),*xy(xx+ex,yy)),fill=ink,width=3)
    draw.line((*xy(xx,yy-ey),*xy(xx,yy+ey)),fill=ink,width=3)
    draw.ellipse((px-7,py-7,px+7,py+7),fill=blue)
draw.text((240,215),f'Fit through origin: m = {m:.4f} +/- {sigma_m:.4f}',fill=blue,font=f)
xy=panel(980,1310,-.04,.04,'Residual: sin(theta_in) - m sin(theta_out)')
draw.line((*xy(-.02,0),*xy(.72,0)),fill=blue,width=3)
for xx,yy,ee in zip(x,residual,1/np.sqrt(w)):
    px,py=xy(xx,yy)
    draw.line((*xy(xx,yy-ee),*xy(xx,yy+ee)),fill=ink,width=3)
    draw.ellipse((px-7,py-7,px+7,py+7),fill=blue)
draw.text((650,1370),'sin(theta_out)',fill=ink,font=f)
draw.text((170,1440),'Error bars use the stated angular-uncertainty model.',fill=ink,font=font(25))
im.save(root/'part4-fit.png')
