import soundfile as sf
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os.path
import librosa
import random
import shutil

import torch

import time
import math
import glob
import sys
import os

from lib.model_param_init import ModelParameters
from main import inference
from lib import spec_utils

def crop(a,b, isMono=False):
    l = min([a[0].size, b[0].size])
    if isMono:
        return a[:l], b[:l]
    else:
        return a[:l,:l], b[:l,:l]

def ensembleIteration(count):
    count -= 1
    count_temp = count
    result = 0
    for i in range(count):
        i = count_temp - i
        result += i
    return result

def ens_tlv(hl, n_fft, o, inp, algorithm='invert',supress=False):
    if algorithm == 'invert':
        #print('using ALGORITHM: INVERT')
        v_spec_m = inp[0] - inp[1]
    if algorithm == 'comb_norm': # debug
        v_spec_m = inp[0] + inp[1]
        v_spec_m = v_spec_m / 2
    w1,_ = librosa.load(inp[0], sr=44100, mono=False, res_type='polyphase')
    w2,_ = librosa.load(inp[1], sr=44100, mono=False, res_type='polyphase')
    w1,w2=crop(w1,w2)
    t1 = spec_utils.wave_to_spectrogram(w1, hl, n_fft, False, False)
    t2 = spec_utils.wave_to_spectrogram(w2, hl, n_fft, False, False)
    if algorithm == 'min_mag':
        #print('using ALGORITHM: MIN_MAG')
        v_spec_m = np.where(np.abs(t1) <= np.abs(t2), t1, t2)
    if algorithm == 'max_mag':
        #print('using ALGORITHM: MAX_MAG')
        v_spec_m = np.where(np.abs(t1) >= np.abs(t2), t1, t2)
    sf.write('{}.wav'.format(o), spec_utils.spectrogram_to_wave(v_spec_m, hl, False, False).T, 44100)

def whatParameterDoIUseForThisModel(modelname):
    if '4band' in modelname:
        parameter = 'modelparams/4band_44100.json'
    elif '3Band' in modelname:
        parameter = 'modelparams/3band_44100.json'
    elif 'MIDSIDE' in modelname:
        parameter = 'modelparams/3band_44100_mid.json'
    elif '2Band' in modelname:
        parameter = 'modelparams/2band_48000.json'
    elif 'LOFI' in modelname:
        parameter = 'modelparams/2band_44100_lofi.json'
    else:
        parameter = 'modelparams/1band_sr44100_hl512.json'
    return parameter

def whatArchitectureIsThisModel(modelname):
        if 'arch-default' in modelname:
            return 'default'
        elif 'arch-34m' in modelname:
            return '33966KB'
            #from lib import nets_33966KB as nets 
        elif 'arch-124m' in modelname:
            return '123821KB'
            #from lib import nets_123821KB as nets
        elif 'arch-130m' in modelname:
            return '129605KB'
            #from lib import nets_129605KB as nets
        elif 'arch-500m' in modelname:
            return '537238KB'
        else:
            print('Error! autoDetect_arch. Did you modify model filenames?')
            return 'default'

def ensemble(input,model=[],suffix=''):
    if len(model) <= 1:
        print('You need at least 2 models!')
        sys.exit()
    if args.temp[len(args.temp)-1] == '/' or args.temp[len(args.temp)-1] == '\\':
        savedtemp = args.temp[:len(args.temp)-1]
    else:
        savedtemp = args.temp
    print('Temporary files will be saved in {}/*'.format(savedtemp))
    if os.path.isdir('{}/stage_{}'.format(savedtemp,len(model)+1)):
        pass
    else:
        # just to make sure, for past conversion
        for i in glob.glob(f'{savedtemp}/*'):
            shutil.rmtree(i)
        for stages in range(1,int(len(model))+1):
            os.makedirs(rf'{savedtemp}/stage_{stages}')
        os.makedirs(rf'{savedtemp}/temp')
    print('Processing {} with {} models...'.format(os.path.splitext(os.path.basename(input))[0],len(model)))
    r = 0
    print('---------------------------------------------------------')
    for models in model:
        print('USING MODEL: {}'.format(os.path.splitext(os.path.basename(models))[0]))
        model_params = whatParameterDoIUseForThisModel(models)
        arch = whatArchitectureIsThisModel(models)
        isVocalModel = False
        if args.flipVocals:
            if 'vocal' in models.lower():
                isVocalModel = True
            else:
                isVocalModel = False
        process = inference(input,
                            model_params,
                            models,
                            gpu=0,
                            hep=args.high_end_process,
                            wsize=args.window_size,
                            agr=args.aggressiveness,
                            tta=args.tta, # rip if this is true lol
                            oi=False,
                            de=args.deepextraction,
                            pp=args.postprocess,
                            fn=r,
                            spth=f'{savedtemp}/temp',
                            v=isVocalModel,
                            arch=arch)
        process.inference()
        r += 1
        print('---------------------------------------------------------')
    modelname = []
    for i in model:
        modelname.append(os.path.splitext(os.path.basename(i))[0])
    print('Ensembling Instruments...')
    iter = ensembleIteration(r)
    progress_bar = tqdm(total=iter)

    ins_algo = 'min_mag'
    voc_algo = 'max_mag'
    n_fft = 2048
    hl = 512
    for a in range(0,r):
        if a + 1 > r-1:
            break
        ens_tlv(hl, n_fft, f'{savedtemp}/stage_1/{a}',[f'{savedtemp}/temp/{a}_{modelname[a]}_Instruments.wav',f'{savedtemp}/temp/{a+1}_{modelname[a+1]}_Instruments.wav'], algorithm=ins_algo,supress=True)
        progress_bar.update(1)
    # set 0
    for folder in range(2,r):
        for a in range(r-folder):
            ens_tlv(hl, n_fft, f'{savedtemp}/stage_{folder}/{a}',[f'{savedtemp}/stage_{folder-1}/{a}.wav',f'{savedtemp}/stage_{folder-1}/{a+1}.wav'], algorithm=ins_algo,supress=True)
            progress_bar.update(1)
    if os.path.isfile(f'separated/{os.path.splitext(os.path.basename(input))[0]}_Ensembled_Instruments.wav'):
        os.remove(f'separated/{os.path.splitext(os.path.basename(input))[0]}_Ensembled_Instruments.wav')
    bsnme = os.path.splitext(os.path.basename(input))[0]
    final_ens = os.path.join(f'{savedtemp}/stage_{r-1}/','0.wav')
    rename = os.path.join(f'{savedtemp}/stage_{r-1}/',f'{bsnme}_Ensembled_Instruments.wav')
    os.rename(final_ens,rename)
    final_ens = f'{savedtemp}/stage_{r-1}/{bsnme}_Ensembled_Instruments.wav'
    if os.path.isfile(f'separated/{bsnme}'):
        final_ens = f'{savedtemp}/stage_{r-1}/{random.randint(0,22)}_{bsnme}_Ensembled_Instruments.wav'
    shutil.move(final_ens,'separated/')
    progress_bar.close()

    print('Ensembling Vocals...')
    iter = ensembleIteration(r)
    progress_bar = tqdm(total=iter)
    for a in range(0,r):
        if a + 1 > r-1:
            break
        ens_tlv(hl, n_fft, f'{savedtemp}/stage_1/{a}',[f'{savedtemp}/temp/{a}_{modelname[a]}_Vocals.wav',f'{savedtemp}/temp/{a+1}_{modelname[a+1]}_Vocals.wav'], algorithm=voc_algo,supress=True)
        progress_bar.update(1)
    # set 0
    for folder in range(2,r):
        for a in range(r-folder):
            ens_tlv(hl, n_fft, f'{savedtemp}/stage_{folder}/{a}',[f'{savedtemp}/stage_{folder-1}/{a}.wav',f'{savedtemp}/stage_{folder-1}/{a+1}.wav'], algorithm=voc_algo,supress=True)
            progress_bar.update(1)
    if os.path.isfile(f'separated/{os.path.splitext(os.path.basename(input))[0]}_Ensembled_Vocals.wav'):
        os.remove(f'separated/{os.path.splitext(os.path.basename(input))[0]}_Ensembled_Vocals.wav')
    bsnme = os.path.splitext(os.path.basename(input))[0]
    final_ens = os.path.join(f'{savedtemp}/stage_{r-1}/','0.wav')
    rename = os.path.join(f'{savedtemp}/stage_{r-1}/',f'{bsnme}_Ensembled_Vocals.wav')
    os.rename(final_ens,rename)
    final_ens = f'{savedtemp}/stage_{r-1}/{bsnme}_Ensembled_Vocals.wav'
    if os.path.isfile(f'separated/{bsnme}'):
        final_ens = f'{savedtemp}/stage_{r-1}/{random.randint(0,22)}_{bsnme}_Ensembled_Vocals.wav'
    shutil.move(final_ens,'separated/')
    progress_bar.close()

p = argparse.ArgumentParser()
p.add_argument('--input', '-i', required=True, help='Input')
p.add_argument('--model_ens', nargs='+', required=False, default=[], help='Ensemble the models of your choice') # MODEL RESULT ENSEMBLING
p.add_argument('--temp','-T', help='temp file location',default='/content/temp')
p.add_argument('--suppress', '-s', action='store_true', help='Hide Warnings')
p.add_argument('--start', default=1)
p.add_argument('--stop', default=1)
p.add_argument('--increments', default=1)

p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'mirroring', 'mirroring2'], default='none', help='Working with extending a low band model.')
p.add_argument('--window_size', '-w', type=int, default=512, help='Window size')
p.add_argument('--aggressiveness', '-A', type=float, default=0.07, help='Aggressiveness of separation')
p.add_argument('--tta', '-t', action='store_true', help='Perform Test-Time-Augmentation to improve the separation quality.')
p.add_argument('--deepextraction', '-D', action='store_true', help='Deeply remove vocals from instruments')
p.add_argument('--postprocess', '-p', action='store_true', help='Masks instrumental part based on the vocals volume.')

p.add_argument('--flipVocals','-v',action='store_true',help='Automatically flip vocal model stems')

args = p.parse_args()
#----------------------------------
if args.suppress:
    warnings.filterwarnings("ignore")

# ---------------------------------
ensemble(args.input,args.model_ens)