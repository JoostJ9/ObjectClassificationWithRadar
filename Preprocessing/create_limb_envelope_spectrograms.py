import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import multiprocessing as mp

# --- Helper functions ---

def iq_correction(raw_data):
    """
    Perform I/Q correction on complex radar data.
    Removes DC offset, and corrects for amplitude and phase imbalance.
    """
    i_data = np.real(raw_data)
    q_data = np.imag(raw_data)
    
    i_dc = np.mean(i_data)
    q_dc = np.mean(q_data)
    i_data = i_data - i_dc
    q_data = q_data - q_dc
    
    i_amp = np.sqrt(np.mean(i_data**2))
    q_amp = np.sqrt(np.mean(q_data**2))
    
    # Add robustness for silent signals to prevent division by zero
    if i_amp == 0 or q_amp == 0:
        return i_data + 1j * q_data
        
    amp_correction = np.sqrt(i_amp * q_amp)
    i_data = i_data * (amp_correction / i_amp)
    q_data = q_data * (amp_correction / q_amp)
    
    iq_corr = np.mean(i_data * q_data)
    # Clamp the argument to arcsin to avoid domain errors from float precision
    arg = np.clip(iq_corr / (i_amp * q_amp), -1.0, 1.0)
    phase_error = np.arcsin(arg)
    q_data_corr = q_data * np.cos(phase_error) - i_data * np.sin(phase_error)
    
    return i_data + 1j * q_data_corr

def kalman_filter_1d(observed, dt, process_noise, measurement_noise):
    """
    Applies a 1D Kalman filter to smooth a sequence of observations.
    """
    A = np.array([[1, dt], [0, 1]])
    C = np.array([[1, 0]])
    Q = process_noise * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    R = measurement_noise ** 2

    x = np.array([[observed[0]], [0]])
    P = np.eye(2)
    filtered = []

    for z in observed:
        x = A @ x
        P = A @ P @ A.T + Q
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        x = x + K @ (z - C @ x)
        P = (np.eye(2) - K @ C) @ P
        filtered.append(x[0, 0])
    
    return np.array(filtered)

def remove_torso_from_spectrogram(Sxx, velocity_axis, torso_velocity_trace, bandwidth=0.4):
    """
    Removes the torso signature from the spectrogram and centers all motion 
    around zero velocity for better limb motion analysis.
    """
    Sxx_centered = np.zeros_like(Sxx)
    zero_idx = np.argmin(np.abs(velocity_axis - 0))

    for i, torso_vel in enumerate(torso_velocity_trace):
        shift_idx = np.argmin(np.abs(velocity_axis - torso_vel))
        shift_amount = zero_idx - shift_idx
        shifted_col = np.roll(Sxx[:, i], shift=shift_amount)
        suppress_mask = np.abs(velocity_axis) <= bandwidth
        shifted_col[suppress_mask] = 0
        Sxx_centered[:, i] = shifted_col

    return Sxx_centered

def detect_envelope(Sxx_dB, velocity_axis, threshold_dB):
    """
    Detects the upper and lower velocity envelopes from a spectrogram.
    """
    n_bins, n_frames = Sxx_dB.shape
    upper_envelope = np.full(n_frames, np.nan)
    lower_envelope = np.full(n_frames, np.nan)

    for i in range(n_frames):
        # Find indices where the signal is above the threshold
        above_thresh = np.where(Sxx_dB[:, i] > threshold_dB)[0]
        if len(above_thresh) > 0:
            # Get the velocities corresponding to the lowest and highest index
            lower_envelope[i] = velocity_axis[above_thresh[0]]
            upper_envelope[i] = velocity_axis[above_thresh[-1]]

    return lower_envelope, upper_envelope


# --- Main processing function (Updated for Limb Envelope) ---

def process_radar_file(file_info):
    """Process a single radar data file and generate a limb envelope spectrogram."""
    file_path, output_dir = file_info
    try:
        print(f"Processing: {file_path}")
        
        # --- Data Reading and Parameter Extraction ---
        with open(file_path, 'r') as f:
            lines = f.readlines()
        header = [float(lines[i].strip()) for i in range(4)]
        data_strs = [line.strip().replace('i', 'j') for line in lines[4:]]
        data_complex = np.array([complex(s) for s in data_strs])
        radarData = np.array(header + list(data_complex))
        fc, Tsweep, NTS, Data = radarData[0], radarData[1]/1000, int(radarData[2]), radarData[4:]

        # --- I/Q Correction, Reshaping, Range-FFT, MTI Filter ---
        Data_corrected = iq_correction(Data)
        nc = int(len(Data_corrected) / NTS)
        Data_time = Data_corrected.reshape((NTS, nc), order='F')
        tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
        Data_range = tmp[NTS//2:, :]
        b, a = signal.butter(4, 0.0075, 'high')
        Data_range_MTI = signal.lfilter(b, a, Data_range, axis=1)
        Data_range_MTI = Data_range_MTI[1:, :]

        # --- Spectrogram Generation ---
        base_filename = Path(file_path).stem
        bin_indl, bin_indu = 9, 29
        PRF = 1 / Tsweep
        TimeWindowLength = 200
        OverlapLength = int(np.round(TimeWindowLength * 0.95))
        FFTPoints = 4 * TimeWindowLength

        Data_spec_MTI2 = None
        for RBin in range(bin_indl, bin_indu + 1):
            f, t, Sxx = signal.spectrogram(Data_range_MTI[RBin, :], fs=PRF, window='hann',
                                           nperseg=TimeWindowLength, noverlap=OverlapLength,
                                           nfft=FFTPoints, mode='complex')
            Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
            if Data_spec_MTI2 is None:
                Data_spec_MTI2 = np.abs(Sxx_shifted)
            else:
                Data_spec_MTI2 += np.abs(Sxx_shifted)
        
        # --- Normalization and Torso Removal ---
        if np.max(Data_spec_MTI2) > 0:
            Data_spec_MTI2 /= np.max(Data_spec_MTI2)
        
        DopplerAxis = np.fft.fftshift(np.fft.fftfreq(FFTPoints, d=1 / PRF))
        velocity_axis = DopplerAxis * 3e8 / (2 * fc)
        
        raw_indices = np.argmax(Data_spec_MTI2, axis=0)
        raw_torso_velocity = velocity_axis[raw_indices]
        
        dt = t[1] - t[0] if len(t) > 1 else 1 / PRF
        kalman_torso_velocity = kalman_filter_1d(raw_torso_velocity, dt, process_noise=10000.0, measurement_noise=0.1)
        
        Sxx_torso_removed = remove_torso_from_spectrogram(Data_spec_MTI2, velocity_axis, kalman_torso_velocity)

        # --- STEP 7: Limb Envelope Detection and Plotting ---
        Sxx_dB = 20 * np.log10(Sxx_torso_removed + 1e-12)

        noise_floor = np.percentile(Sxx_dB, 10)
        threshold_dB = noise_floor + 20

        lower_env, upper_env = detect_envelope(Sxx_dB, velocity_axis, threshold_dB)

        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 5))
        plt.imshow(Sxx_dB, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], velocity_axis[0], velocity_axis[-1]],
                   cmap='inferno', vmin=-100, vmax=0)
        
        plt.plot(t, lower_env, color='lime', linewidth=2, label='Lower Envelope')
        plt.plot(t, upper_env, color='magenta', linewidth=2, label='Upper Envelope')
        
        plt.colorbar(label='Magnitude [dB]')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Limb Envelope Detection (Torso Removed)')
        plt.legend()
        plt.ylim(-6, 6)
        plt.tight_layout()
        
        output_filename = os.path.join(output_dir, f'{base_filename}_limb_envelope.png')
        plt.savefig(output_filename, dpi=150)
        plt.close()
        
        print(f"Successfully processed and saved: {output_filename}")
        return (file_path, True, None)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return (file_path, False, str(e))

def batch_process_radar_data(datasets_root='datasets', output_root='spectrograms_limb_envelope', num_processes=None):
    """
    Process all .dat files to create limb envelope spectrograms.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} CPU cores for processing.")
    os.makedirs(output_root, exist_ok=True)
    
    dat_files = glob.glob(os.path.join(datasets_root, '**', '*.dat'), recursive=True)
    
    if not dat_files:
        print(f"No .dat files found in '{datasets_root}'")
        return
    
    print(f"Found {len(dat_files)} .dat files to process.")
    
    file_info_list = []
    for dat_file in dat_files:
        rel_path = os.path.relpath(dat_file, datasets_root)
        output_subdir = os.path.join(output_root, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)
        file_info_list.append((dat_file, output_subdir))
    
    print(f"Starting parallel processing with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_radar_file, file_info_list)
    
    processed_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - processed_count
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process: {failed_count} files")
    
    if failed_count > 0:
        print("\nFailed files:")
        for file_path, success, error_msg in results:
            if not success:
                print(f"  {file_path}: {error_msg}")
    
    print(f"Output saved to: {output_root}")

if __name__ == "__main__":
    if not os.path.exists('datasets'):
        print("Error: 'datasets' directory not found.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure the 'datasets' directory exists and contains .dat files.")
    else:
        batch_process_radar_data(num_processes=12)