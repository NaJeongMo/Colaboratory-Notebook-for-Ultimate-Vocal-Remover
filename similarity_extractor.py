from lib import spec_utils
from lib import nets
from lib import vr

import soundfile as sf
import numpy as np
import argparse
import os.path
import librosa
import torch
import time
import os

def align_trim(a,b, sr=44100):
    a, _ = librosa.effects.trim(a)
    b, _ = librosa.effects.trim(b)
    a_mono = a[:, :sr * 4].sum(axis=0)
    b_mono = b[:, :sr * 4].sum(axis=0)
    a_mono -= a_mono.mean()
    b_mono -= b_mono.mean()
    offset = len(a_mono) - 1
    delay = np.argmax(np.correlate(a_mono, b_mono, 'full')) - offset
    if delay > 0:
        a = a[:, delay:]
    else:
        b = b[:, np.abs(delay):]
    if a.shape[1] < b.shape[1]:
        b = b[:, :a.shape[1]]
    else:
        a = a[:, :b.shape[1]]
    return a, b
def crop(a,b, isMono=False):
    l = min([a[0].size, b[0].size])
    if isMono:
        return a[:l], b[:l]
    else:
        return a[:l,:l], b[:l,:l]

p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--track1',type=str, required=True)
p.add_argument('--track2',type=str, required=True)
p.add_argument('--wsize',type=int, default=320)
p.add_argument('--agr',type=float, default=0.02)
p.add_argument('--difference', action='store_true')
p.add_argument('--align', action='store_true')
p.add_argument('--model',type=str, default='models/v5_new/ensemble.pth')
p.add_argument('--double', action='store_true')
p.add_argument('--sr', type=int, default=44100) # this doesn't work
args = p.parse_args()
start_time = time.time()
print('loading tracks...', end=' ')
wave1,_ = librosa.load(args.track1, mono=False,sr=args.sr)
wave2,_ = librosa.load(args.track2, mono=False,sr=args.sr)
if args.align:
    wave1,wave2 = align_trim(wave1,wave2, sr=args.sr)
else:
    wave1,wave2 = crop(wave1,wave2)
L1 = wave1[0]
L2 = wave2[0]
R1 = wave1[1]
R2 = wave2[1]
L = np.asfortranarray([L1,L2], np.float32)
R = np.asfortranarray([R1,R2], np.float32)
print('done')
if args.difference == False:
    del wave1, wave2
print('loading model...', end=' ')
device = torch.device('cpu')
model = nets.CascadedASPPNet(2048) # n_fft
model.load_state_dict(torch.load(args.model, map_location=device))
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
vr = vr.VocalRemover(model, device, args.wsize)
print('done')
print()
inputname = os.path.splitext(os.path.basename(args.track1))[0]

for i in range(2):
    if i == 0:
        channel = 'left'
    elif i == 1:
        channel = 'right'
    print(f'loading & stft of {channel} wave source...', end=' ')
    ## X_wave, X_spec_s = {}, {}
    
    if channel == 'left':
        X_spec_m = spec_utils.wave_to_spectrogram_mt(L, 512, 2048, False, False)
    else:
        X_spec_m = spec_utils.wave_to_spectrogram_mt(R, 512, 2048, False, False)
    print('done')
    pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.agr, 'split_bin': 1024})
    y_spec_m = pred * X_phase
    if channel == 'left':
        L_spec_m = y_spec_m
    else:
        R_spec_m = y_spec_m
    print()
if args.double:
    for i in range(2): # finalising
        if i == 0:
            channel = 'left'
        elif i == 1:
            channel = 'right'
        print(f'finalising {channel} channel...')
        if channel == 'left':
            pred, X_mag, X_phase = vr.inference(L_spec_m, {'value': args.agr, 'split_bin': 1024})
        else:
            pred, X_mag, X_phase = vr.inference(R_spec_m, {'value': args.agr, 'split_bin': 1024})
        y_spec_m = pred * X_phase
        if channel == 'left':
            L_spec_m = y_spec_m
        else:
            R_spec_m = y_spec_m
        print()
saveFolder = 'separated'
L_wav = spec_utils.spectrogram_to_wave(L_spec_m, 512, False, False)
R_wav = spec_utils.spectrogram_to_wave(R_spec_m, 512, False, False)
monoL = L_wav[0] + L_wav[1]
monoR = R_wav[0] + R_wav[1]
wave = np.array([monoL/2,monoR/2],np.float32)
#wave = np.array([wave[0],wave[0]], np.float32)
sf.write(os.path.join(saveFolder,'{}_similarity.wav').format(inputname), wave.T, args.sr)
if args.difference:
    wave,wave1=crop(wave,wave1)
    wave1,wave2=crop(wave1,wave2)
    diff1 = wave - wave1
    diff2 = wave - wave2
    sf.write(os.path.join(saveFolder,'{}_difference_1.wav').format(inputname), diff1.T, args.sr)
    sf.write(os.path.join(saveFolder,'{}_difference_2.wav').format(inputname), diff2.T, args.sr)
print('Complete!')
print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

