import numpy as np

def propagate(function, values, errors):
    values = np.array(values, dtype=float)
    errors = np.array(errors, dtype=float)
    answer = np.atleast_1d(function(*values))
    variance = np.zeros(len(answer))

    for i in range(len(values)):
        step = 1e-5 * max(abs(values[i]), 1)
        high = values.copy()
        low = values.copy()
        high[i] += step
        low[i] -= step
        derivative = (
            np.atleast_1d(function(*high))
            - np.atleast_1d(function(*low))
        ) / (2 * step)
        variance += (derivative * errors[i]) ** 2

    return np.sqrt(variance)


screen_position = 95.0

lens_A_intervals = np.array([
    [67.8, 69.0],
    [67.8, 69.1],
    [67.8, 68.5]
])

lens_B_intervals = np.array([
    [73.0, 73.3],
    [72.9, 73.6],
    [73.2, 73.4]
])

lens_C_intervals = np.array([
    [88.9, 89.0],
    [88.9, 89.1],
    [88.9, 89.1]
])

lens_intervals = [lens_A_intervals, lens_B_intervals, lens_C_intervals]

focal_lengths = []
focal_length_means = []
focal_length_sems = []
focus_half_widths = []

for intervals in lens_intervals:
    lens_positions = np.mean(intervals, axis=1)
    focal_values = screen_position - lens_positions
    focal_lengths.append(focal_values)
    focal_length_means.append(np.mean(focal_values))
    focal_length_sems.append(np.std(focal_values, ddof=1) / np.sqrt(len(focal_values)))
    focus_half_widths.append(np.mean((intervals[:, 1] - intervals[:, 0]) / 2))

focal_length_means = np.array(focal_length_means)
focal_length_sems = np.array(focal_length_sems)
focus_half_widths = np.array(focus_half_widths)


x_object_C = 17.4
object_heights = np.array([2.230, 2.228, 2.224])
object_height = np.mean(object_heights)
object_height_error = np.std(object_heights, ddof=1) / np.sqrt(len(object_heights))

lens_C_positions = np.array([27.4, 27.9, 26.9])
screen_intervals_C = np.array([
    [37.2, 38.6],
    [37.6, 37.9],
    [37.0, 38.2]
])
closest_heights = np.array([2.2, 2.2, 2.5])
farthest_heights = np.array([2.8, 2.4, 2.7])

screen_positions_C = np.mean(screen_intervals_C, axis=1)
screen_errors_C = (screen_intervals_C[:, 1] - screen_intervals_C[:, 0]) / 2
image_heights_C = (closest_heights + farthest_heights) / 2
image_height_errors_C = np.full(3, np.sqrt(0.1**2 + 0.2**2) / 2)

do_C = lens_C_positions - x_object_C
di_C = screen_positions_C - lens_C_positions
distance_magnification_C = -di_C / do_C
height_magnification_C = -image_heights_C / object_height
height_magnification_errors_C = abs(height_magnification_C) * np.sqrt(
    (image_height_errors_C / image_heights_C) ** 2
    + (object_height_error / object_height) ** 2
)


x_intermediate = 33.2
lens_B_positions = np.array([63.2, 62.7, 63.7])
final_screen_intervals = np.array([
    [94.6, 96.2],
    [95.9, 97.0],
    [95.7, 96.4]
])

final_screen_positions = np.mean(final_screen_intervals, axis=1)
do_B_real = lens_B_positions - x_intermediate
di_B_real = final_screen_positions - lens_B_positions
f_B_real = do_B_real * di_B_real / (do_B_real + di_B_real)


position_error = 0.05
x_B = 28.55
x_screen = 33.55
x_image_C = 35.92


def experiment_12(xb, xic, xs):
    do = xb - xic
    di = xs - xb
    magnification = -di / do
    focal_length = do * di / (do + di)
    return np.array([do, di, magnification, focal_length])


experiment_12_results = experiment_12(x_B, x_image_C, x_screen)
experiment_12_errors = propagate(
    experiment_12,
    [x_B, x_image_C, x_screen],
    [position_error, position_error, position_error]
)


experiment_13_data = np.array([
    [13.5, 24.1, 49.8, 75.2, 3.1],
    [13.5, 26.9, 55.5, 75.5, 2.7],
    [13.5, 27.9, 55.3, 74.1, 2.6],
    [13.5, 23.3, 48.0, 74.1, 3.2],
    [13.5, 21.0, 40.2, 76.2, 4.7]
])

f_A = 18.0
f_A_error = 0.5
final_height_error = 0.1


def experiment_13(xo, xb, xa, xs, hi, fa, ho):
    do_B = xb - xo
    di_A = xs - xa
    do_A = 1 / (1 / fa - 1 / di_A)
    di_B = (xa - xb) - do_A
    predicted_magnification = abs((di_B / do_B) * (di_A / do_A))
    measured_magnification = hi / ho
    m_A = -di_A / do_A
    m_B = (-measured_magnification) / m_A
    f_B = do_B * di_B / (do_B + di_B)

    return np.array([
        do_B,
        di_A,
        do_A,
        di_B,
        predicted_magnification,
        measured_magnification,
        m_A,
        m_B,
        f_B
    ])


experiment_13_results = []
experiment_13_errors = []

for row in experiment_13_data:
    values = [*row, f_A, object_height]
    errors = [
        position_error,
        position_error,
        position_error,
        position_error,
        final_height_error,
        f_A_error,
        object_height_error
    ]
    experiment_13_results.append(experiment_13(*values))
    experiment_13_errors.append(propagate(experiment_13, values, errors))

experiment_13_results = np.array(experiment_13_results)
experiment_13_errors = np.array(experiment_13_errors)

f_B_values = experiment_13_results[:, 8]
f_B_errors = experiment_13_errors[:, 8]
weights = 1 / f_B_errors**2
f_B_weighted_mean = np.average(f_B_values, weights=weights)
f_B_weighted_error = np.sqrt(1 / np.sum(weights))

near_points_Bryan = np.array([9.7, 10.0, 9.4])
near_points_Ines = np.array([11.0, 10.6, 10.5])

near_point_Bryan = np.mean(near_points_Bryan)
near_point_Ines = np.mean(near_points_Ines)

near_point_error_Bryan = np.std(near_points_Bryan, ddof=1) / np.sqrt(len(near_points_Bryan))
near_point_error_Ines = np.std(near_points_Ines, ddof=1) / np.sqrt(len(near_points_Ines))

f_C = 5.080
f_C_error = 0.041

predicted_magnification_Bryan = near_point_Bryan / f_C
predicted_magnification_Ines = near_point_Ines / f_C

visual_magnification_Bryan = 1.5
visual_magnification_Ines = 2.0

camera_distances = np.array([7.5, 7.3, 7.2])
mean_camera_distance = np.mean(camera_distances)
camera_distance_sem = np.std(camera_distances, ddof=1) / np.sqrt(len(camera_distances))