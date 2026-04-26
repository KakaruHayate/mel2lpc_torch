import matplotlib.pyplot as plt
import torch
import numpy as np


from mel2lpc.utils import plot, plot_spec, load_wav, save_wav
from mel2lpc.mel2lpc_torch import Audio2Mel, Mel2LPC, lpc2wav, PreEmphasis

wav_name = 'wavs/vox_1_0.wav'
sample_rate = 44100
n_fft = 2048
num_mels = 128
hop_length = 512
win_length = 2048
lpc_order = 20
clip_lpc = True
mel_fmin = 40
mel_fmax = 16000
mel_base='e'
lpc_solver='yule_walker'


wav_data = load_wav(wav_name, sample_rate)
wav_data = torch.tensor(wav_data).unsqueeze(0).unsqueeze(1)


# preemph = PreEmphasis(coefficient=0.9375)
# wav_data = preemph(wav_data)
a2w = Audio2Mel(
    sampling_rate=sample_rate, 
    hop_length=hop_length, 
    win_length=win_length, 
    n_fft=n_fft, 
    n_mel_channels=num_mels, 
    mel_fmin=mel_fmin, 
    mel_fmax=mel_fmax, 
    mel_base=mel_base
    )
mel = a2w(wav_data)


m2l = Mel2LPC(
    sampling_rate=sample_rate, 
    hop_length=hop_length, 
    win_length=win_length, 
    n_fft=n_fft, 
    n_mel_channels=num_mels, 
    mel_fmin=mel_fmin, 
    mel_fmax=mel_fmax, 
    lpc_order=lpc_order, 
    mel_base=mel_base, 
    lpc_solver=lpc_solver
    )
LPC_ctrl_mel = m2l(mel.transpose(1, 2))


wav_pred_mel = lpc2wav(LPC_ctrl_mel, wav_data, lpc_order=lpc_order, clip_lpc=clip_lpc)


# Make sure the predicted audio and the original audio have the same shape
if wav_pred_mel.shape[2] > wav_data.shape[2]:
    wav_pred_mel = wav_pred_mel[:, :, :wav_data.shape[2]]
elif wav_pred_mel.shape[2] < wav_data.shape[2]:
    wav_data = wav_data[:, :, :wav_pred_mel.shape[2]]


# Compute the residual
residual = wav_data - wav_pred_mel


wav_data = wav_data.squeeze(0).squeeze(0).numpy()
wav_pred_mel = wav_pred_mel.squeeze(0).squeeze(0).numpy()
error = residual.squeeze(0).squeeze(0).numpy()


save_wav(wav_pred_mel, 'wavs/pred.wav', sample_rate)
save_wav(error, 'wavs/error.wav', sample_rate)


fig = plt.figure(figsize=(30, 5))
plt.subplot(311)
plt.ylabel('wav_data')
plt.xlabel('time')
plt.plot(wav_data)
plt.subplot(312)
plt.ylabel('wav_pred_mel')
plt.xlabel('time')
plt.plot(wav_pred_mel)
plt.subplot(313)
plt.ylabel('error')
plt.xlabel('time')
plt.plot(error)
plt.show()
