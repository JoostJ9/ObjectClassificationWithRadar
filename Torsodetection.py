import numpy as np
# Kalman filter for 1D position and velocity estimation
def kalman_filter_1d(observed, dt, process_noise=20.0, measurement_noise=1.0):
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    C = np.array([[1, 0]])
    Q = process_noise * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    R = measurement_noise ** 2

    x = np.array([[observed[0]], [0]])  # initial state: position + velocity
    P = np.eye(2)
    filtered = []

    for z in observed:
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q

        # Update
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        x = x + K @ (z - C @ x)
        P = (np.eye(2) - K @ C) @ P

        filtered.append(x[0, 0])
    
    return np.array(filtered)

def remove_torso_from_spectrogram(Sxx, velocity_axis, torso_velocity_trace, bandwidth=0.4):
    """
    Removes the torso signature from the spectrogram by:
    1. Shifting torso velocity to zero
    2. Zeroing out a band ±bandwidth [m/s] around 0 velocity
    """
    Sxx_torso_removed = np.zeros_like(Sxx)
    zero_idx = np.argmin(np.abs(velocity_axis - 0))  # Only compute once

    for i, torso_vel in enumerate(torso_velocity_trace):
        # Shift to center torso at 0
        shift_idx = np.argmin(np.abs(velocity_axis - torso_vel))
        shift_amount = zero_idx - shift_idx
        shifted_col = np.roll(Sxx[:, i], shift=shift_amount)

        # Apply suppression mask around 0 velocity
        suppress_mask = np.abs(velocity_axis) <= bandwidth
        shifted_col[suppress_mask] = 0

        Sxx_torso_removed[:, i] = shifted_col

    return Sxx_torso_removed

def detect_envelope(Sxx_dB, velocity_axis, threshold_dB):
    n_bins, n_frames = Sxx_dB.shape
    upper_envelope = np.full(n_frames, np.nan)
    lower_envelope = np.full(n_frames, np.nan)

    for i in range(n_frames):
        above_thresh = np.where(Sxx_dB[:, i] > threshold_dB)[0]
        if len(above_thresh) > 0:
            lower_envelope[i] = velocity_axis[above_thresh[0]]
            upper_envelope[i] = velocity_axis[above_thresh[-1]]

    return lower_envelope, upper_envelope