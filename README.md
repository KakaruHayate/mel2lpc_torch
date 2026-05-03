# mel2lpc

A PyTorch implementation that converts mel spectrograms into Linear Predictive Coding (LPC) coefficients and reconstructs or analyses speech signals. This tool is designed for speech processing research, including neural vocoding, waveform modelling, and low‑bitrate coding experiments.

## Features

- **Mel extraction** – `Audio2Mel` computes log‑mel spectrograms with support for keyshift and speed change.
- **Pre‑emphasis** – Optional pre‑emphasis filtering via `PreEmphasis`.
- **Mel‑to‑LPC** – `Mel2LPC` maps mel spectrograms back to the linear frequency domain, computes the autocorrelation via IFFT, applies a lag window, and solves for LPC coefficients with your choice of solver:
  - Levinson‑Durbin recursion
  - Yule‑Walker equations (batch matrix solve with optional regularisation)
- **LPC synthesis** – `lpc2wav` applies the time‑varying all‑pole synthesis filter frame‑by‑frame to predict speech from past samples.
- **Residual extraction** – The example script demonstrates how to compute and save the prediction error (residual).

## File Structure

```
.
├── mel2lpc_torch.py    # Core modules (PreEmphasis, Audio2Mel, Mel2LPC, lpc2wav)
├── utils.py            # Audio I/O and plotting helpers
└── predict.py          # End‑to‑end example: wav → mel → LPC → reconstructed wav
```

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9 (tested with 1.12+)
- numpy
- scipy
- librosa
- matplotlib

Install the dependencies:

```bash
pip install torch numpy scipy librosa matplotlib
```

Clone the repository:

```bash
git clone https://github.com/your-username/mel2lpc.git
cd mel2lpc
```

## Quick Start

1. Place a 16‑bit PCM WAV file inside a `wavs/` folder (create it if needed).  
   Default example: `wavs/vox_1_0.wav`
2. Run the prediction script:

```bash
python predict.py
```

The script will:
- Load the waveform (expected sample rate: 44100 Hz)
- Compute the log‑mel spectrogram
- Estimate LPC coefficients from the mel spectrogram
- Reconstruct the waveform via the LPC synthesis filter
- Save `wavs/pred.wav` (reconstructed) and `wavs/error.wav` (residual)
- Display a plot of the original, predicted, and error signals

## Usage as a Library

You can import the modules into your own projects:

```python
import torch
from mel2lpc_torch import Audio2Mel, Mel2LPC, lpc2wav, PreEmphasis

# --- Audio to Mel ---
a2m = Audio2Mel(
    sampling_rate=44100, hop_length=512, win_length=2048,
    n_fft=2048, n_mel_channels=128, mel_fmin=40, mel_fmax=16000,
    mel_base='e'
)
audio = torch.randn(1, 1, 44100)          # [B, 1, T]
mel   = a2m(audio)                        # [B, T, 128]

# --- Mel to LPC ---
m2l = Mel2LPC(
    sampling_rate=44100, hop_length=512, win_length=2048,
    n_fft=2048, n_mel_channels=128, mel_fmin=40, mel_fmax=16000,
    lpc_order=20, mel_base='e', lpc_solver='yule_walker', lpc_reg=1e-4
)
lpc  = m2l(mel.transpose(1, 2))           # [B, lpc_order, T]

# --- LPC synthesis (prediction) ---
pred = lpc2wav(lpc, audio, lpc_order=20, clip_lpc=True)  # [B, 1, T]
# (audio must have proper history – lpc2wav pads it automatically)
```

## Key Parameters

### `Audio2Mel`
| Parameter | Description | Default |
|-----------|-------------|---------|
| `sampling_rate` | Audio sample rate (Hz) | 44100 |
| `hop_length` | STFT hop size | 512 |
| `win_length` | STFT window length | 2048 |
| `n_fft` | FFT points (defaults to `win_length`) | 2048 |
| `n_mel_channels` | Number of mel bands | 128 |
| `mel_fmin` | Lowest mel frequency (Hz) | 0 |
| `mel_fmax` | Highest mel frequency (Hz) | None (Nyquist) |
| `mel_base` | Logarithm base (`'e'`, `'10'`, `'ori'`) | `'e'` |
| `clamp` | Floor value before log | 1e-5 |

`forward` accepts optional `keyshift` (semitones) and `speed` (time‑stretch factor).

### `Mel2LPC`
| Parameter | Description | Default |
|-----------|-------------|---------|
| `lpc_order` | Order of the LPC filter | 20 |
| `lpc_solver` | Algorithm for LPC estimation (`'levinson'` or `'yule_walker'`) | `'yule_walker'` |
| `lpc_reg` | Regularisation for Yule‑Walker solver | 1e-4 |
| `f0` | Reference F0 for lag window (Hz) | 40 |
| `repeat` | Temporal upsample factor (defaults to `hop_length`) | `hop_length` |

The module reconstructs the linear power spectrum from the mel spectrogram using a pseudo‑inverse of the mel filter bank (`inv_mel_basis`), then obtains the autocorrelation via `IFFT(linear²)`, applies a Gaussian lag window, and finally solves the Yule‑Walker or Levinson‑Durbin equations in a batched, differentiable manner.

### `lpc2wav`
Synthesises speech by filtering the input waveform with the time‑varying LPC coefficients.  
- The input `wav` tensor should have shape `[B, C, T]` (or `[B, C, T + lpc_order]` if history is already included).
- `clip_lpc` optionally clamps the output to `[-1, 1]`.

## Background

LPC coefficients are widely used in speech coding and synthesis. Converting mel spectrograms – a compact, perceptually motivated representation – back to LPC enables efficient waveform generation that can be integrated into neural networks. This project provides differentiable torch implementations of the necessary signal processing steps, making it suitable for training end‑to‑end models.

The lag window (`exp(-0.5 * (2πf₀/fs)² * k²)`) attenuates higher autocorrelation lags, improving numerical stability and adherence to the spectral envelope.
