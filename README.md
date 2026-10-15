# PA-CSI: Wi-Fi Sensing for Human Activity Recognition

PA-CSI performs human activity recognition from Wi-Fi channel state information (CSI), using amplitude and phase features with attention mechanisms and multi-scale convolutional neural networks.

The method is described in [Enhanced Human Activity Recognition Using Wi-Fi Sensing: Leveraging Phase and Amplitude with Attention Mechanisms](https://doi.org/10.3390/s25041038).

## Repository layout

```text
.
├── src/
│   └── pa_csi/           # Model, data, training, and evaluation modules
├── datasets/             # Local data only; not tracked by Git
├── requirements.txt
└── pyproject.toml
```

## Setup

Use Python 3.10, create and activate a virtual environment, then install the project dependencies and package from the repository root:

```console
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The project uses TensorFlow and PyTorch. GPU support depends on compatible CUDA, driver, and framework versions for your operating system.

## Data

Download the original data from [MultiEnv](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments) or [StanWiFi](https://github.com/ermongroup/Wifi_Activity_Recognition). Processed data is also available from the [PA-CSI data folder](https://drive.google.com/drive/folders/1fiJBDWDC3WKkD5pLYRwpLCWMXPVzCQ-i?usp=sharing).

Keep downloaded datasets and generated arrays in local, Git-ignored directories. The default training command expects these files under `datasets/`:

```text
datasets/
├── our_data_amp_1000_270_200.npy
├── our_data_phase_1000_270_200.npy
└── our_data_label_1000_270_200.npy
```

## Train

Run from the repository root:

```console
python -m pa_csi.train
```

## Cross-dataset evaluation

Provide the amplitude, phase, and label arrays for both training and test datasets:

```console
python -m pa_csi.cross_eval --train-amp datasets/train_amp.npy --train-phase datasets/train_phase.npy --train-label datasets/train_label.npy --test-amp datasets/test_amp.npy --test-phase datasets/test_phase.npy --test-label datasets/test_label.npy
```

Use `--epochs` and `--batch-size` to change the evaluation training settings. Run `python -m pa_csi.cross_eval --help` for all options.

## Citation

```bibtex
@article{quy2025enhanced,
  title={Enhanced Human Activity Recognition Using Wi-Fi Sensing: Leveraging Phase and Amplitude with Attention Mechanisms},
  author={Quy, Thai Duy and Lin, Chih-Yang and Shih, Timothy K.},
  year={2025},
  doi={10.3390/s25041038}
}
```
