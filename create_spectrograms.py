import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import multiprocessing as mp
from functools import partial

def process_radar_file(file_info):
    """Process a single radar data file and generate spectrograms"""
    file_path, output_dir = file_info
    try:
        print(f"Processing: {file_path}")
        
        # Read the data from the file as text, replace 'i' with 'j', and convert to complex
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Convert header (first 4 lines) to float
        header = [float(lines[i].strip()) for i in range(4)]

        # Convert the rest to complex numbers
        data_strs = [line.strip().replace('i', 'j') for line in lines[4:]]
        data_complex = np.array([complex(s) for s in data_strs])

        # Combine header and data for compatibility with the rest of the code
        radarData = np.array(header + list(data_complex))

        # Extract parameters
        fc = radarData[0]  # Center frequency in Hz
        Tsweep = radarData[1] / 1000  # Sweep time in seconds (converted from ms)
        NTS = int(radarData[2])  # Number of time samples per sweep
        Bw = radarData[3]  # Bandwidth in Hz
        Data = radarData[4:]  # Raw radar data

        # Calculate sampling parameters
        fs = NTS / Tsweep  # Sampling frequency
        nc = int(len(Data) / NTS)  # Number of chirps

        # Reshape data into [NTS, nc] array
        Data_time = Data.reshape((NTS, nc), order='F')  # Fortran order to match MATLAB's column-major

        # Apply a rectangular window
        win = np.ones((NTS, nc))

        # Compute FFT and shift zero-frequency component to center
        tmp = np.fft.fftshift(np.fft.fft(Data_time * win, axis=0), axes=0)
        Data_range = tmp[NTS//2:, :]  # Take positive frequency components

        # Number of sweeps to process (assuming ns = nc)
        ns = nc

        # Design a 4th-order high-pass Butterworth filter
        b, a = signal.butter(4, 0.0075, 'high')

        # Apply filter to each range bin across chirps
        Data_range_MTI = np.zeros_like(Data_range, dtype=complex)
        for k in range(Data_range.shape[0]):
            Data_range_MTI[k, :] = signal.lfilter(b, a, Data_range[k, :])

        # Remove the first range bin
        Data_range_MTI = Data_range_MTI[1:, :]

        # Create base filename without extension
        base_filename = Path(file_path).stem

        # Spectrogram parameters for Doppler processing
        bin_indl = 10 - 1  # Adjust for 0-based indexing
        bin_indu = 30 - 1  # Adjust for 0-based indexing
        PRF = 1 / Tsweep  # Pulse repetition frequency
        TimeWindowLength = 200
        OverlapFactor = 0.95
        OverlapLength = int(np.round(TimeWindowLength * OverlapFactor))
        Pad_Factor = 4
        FFTPoints = Pad_Factor * TimeWindowLength

        # Compute spectrogram for selected range bins and sum magnitudes
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

        # Compute Doppler and velocity axes
        DopplerAxis = np.fft.fftshift(np.fft.fftfreq(FFTPoints, d=1 / PRF))
        velocity_axis = DopplerAxis * 3e8 / (2 * fc)  # Convert Doppler frequency to velocity

        # Plot Spectrogram - Use Agg backend for thread safety
        plt.switch_backend('Agg')
        plt.figure(figsize=(12, 8))
        im = plt.imshow(20 * np.log10(Data_spec_MTI2), aspect='auto', origin='lower',
                        extent=[t[0], t[-1], velocity_axis[0], velocity_axis[-1]], cmap='jet')
        plt.ylim(-6, 6)  # Limit velocity axis
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        clim = im.get_clim()
        plt.clim(clim[1] - 40, clim[1])  # Set color limits
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_spectrogram.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully processed: {file_path}")
        return (file_path, True, None)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return (file_path, False, str(e))

def batch_process_radar_data(datasets_root='datasets', output_root='spectrograms', num_processes=None):
    """
    Process all .dat files in the datasets directory and create spectrograms
    maintaining the same directory structure using multiprocessing
    """
    
    # Use all available CPU cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} CPU cores for processing")
    
    # Create the output root directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    
    # Find all .dat files in the datasets directory
    dat_files = glob.glob(os.path.join(datasets_root, '**', '*.dat'), recursive=True)
    
    if not dat_files:
        print(f"No .dat files found in {datasets_root}")
        return
    
    print(f"Found {len(dat_files)} .dat files to process")
    
    # Prepare file information tuples for multiprocessing
    file_info_list = []
    for dat_file in dat_files:
        # Get the relative path from the datasets root
        rel_path = os.path.relpath(dat_file, datasets_root)
        
        # Create the corresponding output directory structure
        output_subdir = os.path.join(output_root, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)
        
        file_info_list.append((dat_file, output_subdir))
    
    # Process files in parallel
    print(f"Starting parallel processing with {num_processes} processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_radar_file, file_info_list)
    
    # Count successful and failed processing
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    for file_path, success, error_msg in results:
        if success:
            processed_count += 1
        else:
            failed_count += 1
            failed_files.append((file_path, error_msg))
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process: {failed_count} files")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error_msg in failed_files:
            print(f"  {file_path}: {error_msg}")
    
    print(f"Output saved to: {output_root}")

# Main execution
if __name__ == "__main__":
    # Check if datasets directory exists
    if not os.path.exists('datasets'):
        print("Error: 'datasets' directory not found in current working directory")
        print(f"Current working directory: {os.getcwd()}")
        print("Please make sure the 'datasets' directory exists and contains .dat files")
    else:
        # Start batch processing with all 12 CPU cores
        batch_process_radar_data(num_processes=12)