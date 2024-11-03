import os
import yaml
import json
import argparse
import pywt
import numpy as np
from femo.data.pipeline import Pipeline
from scipy.signal import decimate
from scipy.interpolate import interp1d
from tqdm import tqdm
from femo.logger import LOGGER

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to data manifest json file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to directory containing .dat and .csv files")
    parser.add_argument("--dataset-name", type=str, default="femo_dataset", help="Dataset name to use for saving")
    parser.add_argument("--config-dir", type=str, default=None, help="Path to configuration directory")
    parser.add_argument("--work-dir", type=str, default="./work_dir", help="Path to save generated artifacts")
    args = parser.parse_args()

    return args


def resample_signal(signal, target_len, freq=1024):

    resampling_factor = target_len / len(signal)
    resampled_freq = freq * resampling_factor

    original_len = len(signal)
    duration = len(signal) / freq
    # Define the time points for the original and target signals
    original_time = np.linspace(0, duration, original_len)  # Original time points
    target_time = np.linspace(0, duration, target_len)      # Target time points for resampling

    # Perform cubic spline interpolation
    interpolator = interp1d(original_time, signal, kind='cubic')
    resampled_signal = interpolator(target_time)

    return resampled_signal, int(resampled_freq)


def downsample_signal(signal, original_freq=1024, target_freq=256):
    # Calculate the downsampling factor
    downsample_factor = original_freq // target_freq
    # Downsample the signal with an anti-aliasing filter
    downsampled_signal = decimate(signal, downsample_factor, ftype='iir')
    return downsampled_signal


def calculate_wavelet_dataset(dataset: list, orig_freq=1024, target_len=512, downsampling_factor=4, scales=np.arange(1, 17)):

    down_freq = orig_freq // downsampling_factor  # Downsampled frequency

    dataset_spectogram = []
    for event in dataset:
        channels = []
        for sensor in range(event.shape[1]):
            signal = event[:, sensor]
            resampled_signal, _ = resample_signal(downsample_signal(signal, orig_freq, down_freq), target_len, down_freq)

            coeffs, _ = pywt.cwt(resampled_signal, scales=scales, wavelet='mexh')
            channels.append(np.swapaxes(coeffs, 0, 1))
    
        dataset_spectogram.append(np.swapaxes(channels, 0, 1))
    
    return np.array(dataset_spectogram)


def main():
    LOGGER.info("Starting feature extraction...")
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.path.join(BASE_DIR, '..', 'configs')
    config_files = ['dataset-cfg.yaml']
    [dataset_cfg] = [yaml.safe_load(open(os.path.join(config_dir, cfg), 'r')) for cfg in config_files]
    pipeline = Pipeline(dataset_cfg.get('data_pipeline'))

    LOGGER.info("Downloading raw input data")

    with open(args.data_manifest, "r") as f:
        data_manifest = json.load(f)

    list_dataset = []
    list_labels = []
    num_classes = 2

    for item in tqdm(data_manifest['items'], desc="Processing items", unit="item"):

        labels = []
        examples = []

        data_file_key = item.get('datFileKey', None)

        data_filename = os.path.join(args.data_dir, data_file_key)

        if not os.path.exists(data_filename):
            raise FileNotFoundError(f"Data file not found at {data_filename}")
        
        extracted_detections = pipeline.process(filename=data_filename, outputs=['extracted_detections'])['extracted_detections']

        for i in range(num_classes):

            if i == 0:
                events = extracted_detections['fp_detections_sensor_data']
            else:
                events = extracted_detections['tp_detections_sensor_data']

            dataset_example = calculate_wavelet_dataset(events)
            examples.append(dataset_example)
            labels.append(float(i) + np.zeros(dataset_example.shape[0]))
        
        list_dataset.append(examples)
        list_labels.append(labels)

    np.save(os.path.join(args.work_dir, args.dataset_name, '_examples.npy'), np.array(list_dataset, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.work_dir, args.dataset_name, '_labels.npy'), np.array(list_labels, dtype=object), allow_pickle=True)
    LOGGER.info(f"Dataset saved to {args.work_dir}")

if __name__ == "__main__":
    main()
