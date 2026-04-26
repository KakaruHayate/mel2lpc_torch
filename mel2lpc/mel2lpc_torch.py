import os
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


class PreEmphasis(torch.nn.Module):
    def __init__(self, coefficient: float = 0.9375):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer('kernel', torch.tensor([-coefficient, 1.], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, signal):
        '''
        Input:
            signal: [B, 1, T]
        Returns:
            signal: [B, 1, T]
        '''
        return F.conv1d(signal, self.kernel, padding=1)


class Audio2Mel(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, clamp=1e-5, mel_base='e'):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.mel_base = mel_base
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.clamp = clamp

        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, 
                                   fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_mel_channels = n_mel_channels
        self.hann_window = {}

    def forward(self, audio, keyshift=0, speed=1):
        '''
        Input:
            audio: [B, 1, T]
        Returns:
            log_mel_spec: [B, T, M] 
        '''
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new,
                         win_length=win_length_new, window=self.hann_window[keyshift_key],
                         center=True, return_complex=True)
        magnitude = torch.abs(fft)
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size-resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
            
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.mel_base == 'ori':
            log_mel_spec = torch.log(1. + 10000 * mel_output) 
        elif self.mel_base == '10':
            log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))
        elif self.mel_base == 'e':
            log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        else:
            raise ValueError("Unsupported mel base type")
        
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)
        log_mel_spec = log_mel_spec.squeeze(2)  # mono -> [B, T, M]

        return log_mel_spec


class Mel2LPC(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, repeat=None, f0=40., 
                 lpc_order=4, clamp=1e-12, mel_base='e',
                 lpc_solver='yule_walker',
                 lpc_reg=1e-4):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        repeat = hop_length if repeat is None else repeat

        self.lpc_solver = lpc_solver
        self.lpc_reg = lpc_reg

        self.mel_base = mel_base
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.repeat = repeat
        self.clamp = clamp
        self.lpc_order = lpc_order

        # get lag_window
        theta = (2 * torch.pi * f0 / self.sampling_rate) ** 2
        self.register_buffer(
            "lag_window",
            torch.exp(-0.5 * theta * torch.arange(self.lpc_order + 1).float() ** 2).unsqueeze(1)
        )  # shape: [lpc_order+1, 1]

        # get inv_mel_basis
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, 
                                   fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        inv_mel_basis = torch.pinverse(mel_basis)
        self.register_buffer("inv_mel_basis", inv_mel_basis)  # [F, M]

    def solve_levinson_durbin(self, pAC):
        """levinson durbin's recursion
        Input:
            pAC: autocorrelation [B, n+1, T]
        Returns:
            lpc: LPC coefficients [B, n, T]
        """
        B, n_plus_1, T = pAC.shape
        n = n_plus_1 - 1
        pLP = torch.zeros(B, n, T, dtype=pAC.dtype, device=pAC.device)
        E = pAC[:, 0, :].clone()  # [B, T]

        for i in range(n):
            if i > 0:
                pAC_slice = pAC[:, 1:i+1, :]
                pAC_rev = torch.flip(pAC_slice, dims=[1])
                sum_term = (pLP[:, :i, :] * pAC_rev).sum(dim=1)
            else:
                sum_term = 0.0

            ki = (pAC[:, i+1, :] + sum_term) / E

            if i > 0:
                old = pLP[:, :i, :].clone()
                pLP[:, :i, :] = old - ki.unsqueeze(1) * torch.flip(old, dims=[1])
            pLP[:, i, :] = -ki

            E = E * (1 - ki * ki).clamp(min=1e-5)

        return pLP

    def solve_yule_walker(self, auto_correlation):
        """
        Yule-Walker matrix solver
        Input:
            auto_correlation: [B, n+1, T]
        Returns:
            lpc_coeffs: [B, n, T]
        """
        B, n_plus_1, T = auto_correlation.shape
        n = n_plus_1 - 1
        if n == 0:
            return torch.zeros(B, 0, T, dtype=auto_correlation.dtype, device=auto_correlation.device)

        device = auto_correlation.device
        dtype = auto_correlation.dtype

        ac = auto_correlation.permute(0, 2, 1)  # [B, T, n+1]

        # Construct Toeplitz matrix R: utilize the lag values of the autocorrelation sequence indexed by |i-j|
        lag = torch.abs(torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None, :])  # [n, n]
        R = ac[:, :, lag]  # [B, T, n, n] Each (i,j) corresponds to r[|i-j|]

        # Diagonal loading (numerical stability)
        eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        R = R + self.lpc_reg * eye

        # Right-end vectors: -r[1], ..., -r[n]
        rhs = -ac[:, :, 1:n+1]  # [B, T, n]

        # Solve Yule-Walker equation in batch
        a = torch.linalg.solve(R, rhs)  # [B, T, n]

        a = a.permute(0, 2, 1)
        return a

    def forward(self, mel):
        '''
        Input:
            mel: [B, M, T]
        Returns:
            lpc_ctrl: [B, lpc_order, T]
        '''
        # 1. mel -> linear spectrum
        if self.mel_base == 'ori':
            mel = (torch.exp(mel) - 1.) / 10000
        elif self.mel_base == 'e':
            mel = torch.exp(mel)
        elif self.mel_base == '10':
            mel = torch.pow(10, mel)
        else:
            raise ValueError("Unsupported mel base type")

        # [F, M] @ [B, M, T] -> [B, F, T]
        linear = torch.clamp_min(
            torch.matmul(self.inv_mel_basis.unsqueeze(0), mel), self.clamp
        )

        # 2. linear -> autocorrelation via IFFT of power spectrum
        power = linear ** 2  # [B, F, T]
        flipped_power = torch.flip(power, dims=[1])[:, 1:-1, :]  # [B, F-2, T]
        fft_power = torch.cat([power, flipped_power], dim=1)     # [B, 2F-2, T] == [B, n_fft, T]
        auto_correlation = torch.fft.ifft(fft_power, dim=1).real  # [B, n_fft, T]

        # 3. apply lag window and keep first (lpc_order+1) coefficients
        auto_correlation = auto_correlation[:, :self.lpc_order + 1, :]  # [B, lpc_order+1, T]
        auto_correlation = auto_correlation * self.lag_window

        # 4. Calculate LPC coefficients based on solver
        if self.lpc_solver == 'levinson':
            lpc_coeffs = self.solve_levinson_durbin(auto_correlation)
        elif self.lpc_solver == 'yule_walker':
            lpc_coeffs = self.solve_yule_walker(auto_correlation)
        else:
            raise ValueError(f"Unsupported LPC solver: {self.lpc_solver}")

        # 5. post-processing
        lpc_ctrl = -1 * torch.flip(lpc_coeffs, dims=[1])

        # 6. temporal repeat (upsampling)
        if self.repeat is not None:
            lpc_ctrl = torch.repeat_interleave(lpc_ctrl, self.repeat, dim=-1)

        return lpc_ctrl


def lpc2wav(lpc_ctrl, wav, lpc_order, clip_lpc):
    '''
    Apply LPC synthesis filter in a frame-by-frame manner.
    Input:
        lpc_ctrl: [B, lpc_order, T]
        wav:      [B, C, T]  (must be padded or pre-pended with history)
        lpc_order: int
        clip_lpc: bool
    Returns:
        pred:     [B, C, T]
    '''
    lpc_ctrl = lpc_ctrl[:, :, :wav.shape[-1]]
    num_points = lpc_ctrl.shape[-1]
    if wav.shape[2] == num_points:
        # prepend lpc_order zeros for initial conditions
        wav = F.pad(wav, (lpc_order, 0))
    elif wav.shape[2] != num_points + lpc_order:
        raise RuntimeError('dimensions of lpcs and audio must match')

    indices = torch.arange(lpc_order).view(-1, 1) + torch.arange(lpc_ctrl.shape[-1])
    signal_slices = wav[:, :, indices]  # [B, C, lpc_order, T]

    # lpc_ctrl: [B, lpc_order, T] -> [B, 1, lpc_order, T]
    pred = torch.sum(lpc_ctrl.unsqueeze(1) * signal_slices, dim=2)  # [B, C, T]

    if clip_lpc:
        pred = torch.clip(pred, -1., 1.)

    return pred