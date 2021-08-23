## type: ignore
""" 
This was supposed to be an importable module,
but I got lazy so if you're planning to change everything here goodluck
""" 
from numpy.core.fromnumeric import take
from pathvalidate import sanitize_filename
import soundfile as sf
from tqdm import tqdm
import numpy as np
import youtube_dl
import importlib
import argparse
import warnings
import os.path
import librosa
import hashlib
import random
import shutil

import torch

import wave
import time
import math
import glob
import cv2
import sys
import os

from lib.model_param_init import ModelParameters
from lib import vr as _inference
from lib import spec_utils


class hide_opt:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
def normalise(wave):
    if max(abs(wave[0])) >= max(abs(wave[1])):
        wave *= 1/max(abs(wave[0]))
    elif max(abs(wave[0])) <= max(abs(wave[1])):
        wave *= 1/max(abs(wave[1]))
    return wave
def take_lowest_val(param, o, inp, algorithm='invert',supress=False):
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    mp = ModelParameters(param)
    for d in range(len(mp.param['band']), 0, -1):
        if supress == False:
            print('Band(s) {}'.format(d), end=' ')
        
        bp = mp.param['band'][d]
                
        if d == len(mp.param['band']): # high-end band
            X_wave[d], _ = librosa.load(
                inp[0], bp['sr'], mono=False, res_type=bp['res_type'])
            y_wave[d], _ = librosa.load(
                inp[1], bp['sr'], mono=False, res_type=bp['res_type'])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
            y_wave[d] = librosa.resample(y_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
        
        X_wave[d], y_wave[d] = spec_utils.align_wave_head_and_tail(X_wave[d], y_wave[d])
        ##wave_to_spectrogram(wave, hop_length, n_fft, mp, multithreading)
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, False)
        y_spec_s[d] = spec_utils.wave_to_spectrogram(y_wave[d], bp['hl'], bp['n_fft'], mp, False) 
        if supress == False:
            print('ok')
    del X_wave, y_wave
    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    y_spec_m = spec_utils.combine_spectrograms(y_spec_s, mp)
    if y_spec_m.shape != X_spec_m.shape:
        print('Warning: The combined spectrograms are different!')   
        print('X_spec_m: ' + str(X_spec_m.shape))
        print('y_spec_m: ' + str(y_spec_m.shape))
    if algorithm == 'invert':
        #print('using ALGORITHM: INVERTB')
        y_spec_m = spec_utils.reduce_vocal_aggressively(X_spec_m, y_spec_m, 0.2)
        v_spec_m = X_spec_m - y_spec_m
    if algorithm == 'min_mag':
        #print('using ALGORITHM: MIN_MAG')
        v_spec_mL = np.where(np.abs(y_spec_m[0]) <= np.abs(X_spec_m[0]), y_spec_m[0], X_spec_m[0])
        v_spec_mR = np.where(np.abs(y_spec_m[1]) <= np.abs(X_spec_m[1]), y_spec_m[1], X_spec_m[1])
        v_spec_m = np.asfortranarray([v_spec_mL,v_spec_mR])
        del v_spec_mL,v_spec_mR
    if algorithm == 'max_mag':
        #print('using ALGORITHM: MAX_MAG')
        v_spec_mL = np.where(np.abs(y_spec_m[0]) >= np.abs(X_spec_m[0]), y_spec_m[0], X_spec_m[0])
        v_spec_mR = np.where(np.abs(y_spec_m[1]) >= np.abs(X_spec_m[1]), y_spec_m[1], X_spec_m[1])
        v_spec_m = np.asfortranarray([v_spec_mL,v_spec_mR])
        del v_spec_mL,v_spec_mR
    if algorithm == 'comb_norm': # debug
        v_spec_m = y_spec_m + X_spec_m
        v_spec_m /= 2
    
    wav = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
    if algorithm == 'comb_norm':
        wav = normalise(wav)
    
    sf.write('{}.wav'.format(o), wav, mp.param['sr'])
    del v_spec_m,y_spec_m,X_spec_m,wav

            

def whatParameterDoIUseForThisModel(modelname):
    modelname = modelname.lower()
    if '4band' in modelname:
        parameter = 'modelparams/4band_44100.json'
    elif '3band' in modelname:
        if 'msb2' in modelname:
            parameter = 'modelparams/3band_44100_msb2.json'
        else:
            parameter = 'modelparams/3band_44100.json'
    elif 'midside' in modelname:
        parameter = 'modelparams/3band_44100_mid.json'
    elif '2band' in modelname:
        parameter = 'modelparams/2band_48000.json'
    elif 'lofi' in modelname:
        parameter = 'modelparams/2band_44100_lofi.json'
    else:
        parameter = 'modelparams/1band_sr44100_hl512.json'
    print(parameter)
    return parameter

class inference:
    def __init__(self, input, param, ptm, gpu=-1, hep='none', wsize=320, agr=0.07, tta=False, oi=False, de=False, v=False, spth='separated', fn='', pp=False, arch='default',
                pp_thres = 0.2, mrange = 32, fsize = 64):
        self.input = input
        self.param = param
        self.ptm = ptm
        self.gpu = gpu
        self.hep = hep
        self.wsize = wsize
        self.agr = agr
        self.tta = tta
        self.oi = oi
        self.de = de
        self.v = v
        self.spth = spth
        self.fn = fn
        self.pp = pp
        self.arch = arch
        self.pp_thres = pp_thres
        self.mrange = mrange
        self.fsize = fsize
    def inference(self):
        nets = importlib.import_module('lib.nets' + f'_{self.arch}'.replace('_default', ''), package=None)
        # load model -------------------------------
        print('loading model...', end=' ')
        mp = ModelParameters(self.param)
        device = torch.device('cpu')
        model = nets.CascadedASPPNet(mp.param['bins'] * 2)
        model.load_state_dict(torch.load(self.ptm, map_location=device))
        if torch.cuda.is_available() and self.gpu >= 0:
            device = torch.device('cuda:{}'.format(self.gpu))
            model.to(device)
        print('done')
        # stft of wave source -------------------------------
        print('stft of wave source...', end=' ')
        X_wave, X_spec_s = {},{}
        if self.fn != '':
            basename = self.fn
        else:
            basename = os.path.splitext(os.path.basename(self.input))[0]
        bands_n = len(mp.param['band'])
        for d in range(bands_n, 0, -1):
            bp = mp.param['band'][d]
            if d == bands_n:
                X_wave[d], _ = librosa.load(
                    self.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
            X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, True) # threading true
            if d == bands_n and self.hep != 'none':
                input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
                input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
        del X_wave, X_spec_s
        print('done')
        # process -------------------------------
        #if 'https://' in args.input:
        #    print('splitting {}'.format(desc['title'])) # desc from YouTube()
        #else:
        #    print('splitting {}'.format(os.path.splitext(os.path.basename(input))[0]))
        vr = _inference.VocalRemover(model, device, self.wsize) # vr module
        if self.tta:
            pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': self.agr, 'split_bin': mp.param['band'][1]['crop_stop']})
        else:
            pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': self.agr, 'split_bin': mp.param['band'][1]['crop_stop']})
        if self.pp:
            print('post processing...', end=' ')
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv, thres=self.pp_thres, min_range=self.mrange, fade_size=self.fsize)
            print('done')
        # swap if v=True
        if self.v:
            stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
        else:
            stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        y_spec_m = pred * X_phase # instruments
        v_spec_m = X_spec_m - y_spec_m # vocals
        if self.hep == 'bypass':
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)
        elif self.hep.startswith('mirroring'):       
            input_high_end_ = spec_utils.mirroring(self.hep, y_spec_m, input_high_end, mp)
            
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)  
        else:
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
        if self.de: # deep extraction
            #print('done')
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), wave, mp.param['sr'])
            print('done')
            wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), wave, mp.param['sr'])
            print('done')
            if self.oi:
                with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(y_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)
                with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(v_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)

            print('Performing Deep Extraction...', end = ' ')
            #take_lowest_val(param, o, inp, algorithm='invert',supress=False)
            if os.path.isdir('/content/tempde') == False:
                os.mkdir('/content/tempde')
            take_lowest_val('modelparams/1band_sr44100_hl512.json',
                            '/content/tempde/difftemp_v',
                            [os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])),os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst']))],
                            algorithm='min_mag',
                            supress=True)
            take_lowest_val('modelparams/1band_sr44100_hl512.json',
                            '/content/tempde/difftemp',
                            ['/content/tempde/difftemp_v.wav',os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst']))],
                            algorithm='invert',
                            supress=True)
            os.rename('/content/tempde/difftemp.wav','/content/tempde/{}_{}_DeepExtraction_Instruments.wav'.format(basename, model_name))
            if os.path.isfile(self.spth+'/{}_{}_DeepExtraction_Instruments.wav'.format(basename, model_name)):
                os.remove(self.spth+'/{}_{}_DeepExtraction_Instruments.wav'.format(basename, model_name))
            shutil.move('/content/tempde/{}_{}_DeepExtraction_Instruments.wav'.format(basename, model_name),self.spth)
            # VOCALS REMNANTS
            if os.path.isfile(self.spth+'/{}_{}_cDeepExtraction_Vocals.wav'.format(basename, model_name)):
                os.remove(self.spth+'/{}_{}_cDeepExtraction_Vocals.wav'.format(basename, model_name))
            excess,_ = librosa.load('/content/tempde/difftemp_v.wav',mono=False,sr=44100)
            _vocal,_ = librosa.load(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])),
                                    mono=False,sr=44100)
            sf.write(self.spth + '/{}_{}_cDeepExtraction_Vocals.wav'.format(basename,model_name),excess.T+_vocal.T,44100)
            print('Complete!')
        else: # args
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), wave, mp.param['sr'])
            print('done')
            if True:
                print('inverse stft of {}...'.format(stems['vocals']), end=' ')
                #v_spec_m = X_spec_m - y_spec_m
                wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
                print('done')
                sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), wave, mp.param['sr'])
            if self.oi:
                with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(y_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)
                with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(v_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)
        torch.cuda.empty_cache() # clear ram <<<
    
    def YouTube(self):
        link = self.input
        inputsha = hashlib.sha1(bytes(link, encoding='utf8')).hexdigest() + '.wav'
        self.input = inputsha
        # 251/140/250/139
        frmt = 'best'
        if 'youtu' in link:
            frmt = '251/140/250/139'
            print('YouTube Link detected')
        else:
            print('Non-YouTube link detected. Attempting to download.')
        opt = {'format': frmt, 'outtmpl': inputsha, 'updatetime': False, 'nocheckcertificate': True}
        print('Downloading...', end=' ')
        with hide_opt():
            with youtube_dl.YoutubeDL(opt) as ydl:
                desc = ydl.extract_info(link, download=not os.path.isfile(inputsha))
        print('done')
        print(desc['title'])
        titlename = desc['title']
        modname = os.path.splitext(os.path.basename(self.ptm))[0]
        inname = os.path.splitext(os.path.basename(inputsha))[0]
        inmodname = inname + '_' + modname
        self.inference()
        os.rename("separated/{}_Instruments.wav".format(inmodname), 'separated/' + sanitize_filename(titlename) + '_{}_Instruments.wav'.format(modname))
        os.rename("separated/{}_Vocals.wav".format(inmodname), 'separated/' + sanitize_filename(titlename) + '_{}_Vocals.wav'.format(modname))
        if self.de:
            os.rename("separated/{}_DeepExtraction_Instruments.wav".format(inmodname), 'separated/' + sanitize_filename(titlename) + '_DeepExtraction_Instruments.wav')
            os.remove(inputsha)
        if os.path.isfile(inputsha):
            os.remove(inputsha)


def whatArchitectureIsThisModel(modelname):
    if 'arch-default' in modelname.lower():
        return 'default'
    elif 'arch-34m' in modelname.lower():
        return '33966KB'
        #from lib import nets_33966KB as nets 
    elif 'arch-124m' in modelname.lower():
        return '123821KB'
        #from lib import nets_123821KB as nets
    elif 'arch-130m' in modelname.lower():
        return '129605KB'
        #from lib import nets_129605KB as nets
    elif 'arch-500m' in modelname.lower():
        return '537238KB'
    else:
        print('Error! autoDetect_arch. Did you modify model filenames?')
        return 'default'

def multi_file(models, inputs):
    print('Multiple files detected.')
    print('---------------------------------------------------------------')
    for model in models:
        for track in inputs:
            print('Now processing: {} \nwith {}'.format(os.path.splitext(os.path.basename(track))[0],os.path.basename(model)))
            model_params = whatParameterDoIUseForThisModel(model)
            arch = whatArchitectureIsThisModel(model)
            process = inference(track, model_params,model,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch,v='vocal' in track.lower(), pp_thres=args.pp_threshold, mrange=args.pp_min_range, fsize=args.pp_fade_size)
            if 'https://' in track:
                process.YouTube()
            else:
                process.inference()
            print('---------------------------------------------------------------')

def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument('--isVocal','-v', action='store_true', help='Flip Instruments and Vocals output (Only for Vocal Models)') # FLIP
    p.add_argument('--output_image', '-I', action='store_true', help='Export Spectogram in an image format')
    p.add_argument('--postprocess', '-p', action='store_true', help='Masks instrumental part based on the vocals volume.')
    p.add_argument('--tta', '-t', action='store_true', help='Perform Test-Time-Augmentation to improve the separation quality.')
    p.add_argument('--suppress', '-s', action='store_true', help='Hide Warnings')

    p.add_argument('--input', '-i', help='Input')
    p.add_argument('--pretrained_model', '-P', type=str, default='', help='Pretrained model')
    p.add_argument('--nn_architecture', '-n', type=str, choices=['default', '33966KB', '123821KB', '129605KB','537238KB'], default='default', help='Model architecture')
    p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'mirroring', 'mirroring2'], default='none', help='Working with extending a low band model.')
    
    p.add_argument('--pp_threshold', '-thres',type=float, default=0.2, help='threshold - This is an argument for post-processing')
    p.add_argument('--pp_min_range', '-mrange',type=int, default=64, help='min_range - This is an argument for post-processing')
    p.add_argument('--pp_fade_size', '-fsize',type=int, default=32, help='fade_size - This is an argument for post-processing')
    
    p.add_argument('--gpu', '-g', type=int, default=-1, help='Use GPU for faster processing')
    p.add_argument('--model_params', '-m', type=str, default='', help="Model's parameter")
    p.add_argument('--window_size', '-w', type=int, default=512, help='Window size')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.07, help='Aggressiveness of separation')

    p.add_argument('--deepextraction', '-D', action='store_true', help='Deeply remove vocals from instruments')
    
    p.add_argument('--convert_all', '-c', action='store_true', help='Split all tracks in tracks/ folder') # ITERATE ALL TRACKS
    p.add_argument('--useAllModel', '-a', type=str, choices=['none', 'v5', 'v5_new', 'all'], default='none', help='Use all models') # ITERATE TO MODEL

    args = p.parse_args()
    if args.suppress:
        warnings.filterwarnings("ignore")
    if args.convert_all or args.useAllModel != 'none':
        # ∕
        if args.convert_all:
            args.input = glob.glob('tracks/*')
        elif type(args.input) == str:
            args.input = [args.input]
        if args.useAllModel == 'v5':
            useModel = glob.glob('models/v5/*.pth')
        elif args.useAllModel == 'v5_new':
            useModel = glob.glob('models/v5_new/*.pth')
        elif args.useAllModel == 'all':
            useModel = glob.glob('models/v5_new/*.pth')
            useModel.extend(glob.glob('models/v5/*.pth'))
        else:
            useModel = glob.glob(args.pretrained_model)
        multi_file(useModel, args.input)

    else:
        arch = args.nn_architecture
        process = inference(args.input,args.model_params,args.pretrained_model,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch,v=args.isVocal, pp_thres=args.pp_threshold, mrange=args.pp_min_range, fsize=args.pp_fade_size)
        if 'https://' in args.input:
            warnings.filterwarnings("ignore")
            process.YouTube()
        else: # single
            process.inference()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
