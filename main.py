## type: ignore
""" 
This was supposed to be an importable module,
but I got lazy so if you're planning to change everything here goodluck
""" 
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

import time
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

def clear_folder(dir):
    for file in os.scandir(dir):
        os.remove(file.path)
def take_lowest_val(param, o, inp, algorithm='invert',supress=False):
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    mp = ModelParameters(param)
    for d in range(len(mp.param['band']), 0, -1):
        if supress == False:
            print('Band(s) {}'.format(d), end=' ')
        
        bp = mp.param['band'][d]
                
        if d == len(mp.param['band']): # high-end band
            X_wave[d], _ = librosa.load(
                inp[0], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
            y_wave[d], _ = librosa.load(
                inp[1], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
            y_wave[d] = librosa.resample(y_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
        
        X_wave[d], y_wave[d] = spec_utils.align_wave_head_and_tail(X_wave[d], y_wave[d])
        
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        y_spec_s[d] = spec_utils.wave_to_spectrogram(y_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse']) 
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
        #print('using ALGORITHM: INVERT')
        y_spec_m = spec_utils.reduce_vocal_aggressively(X_spec_m, y_spec_m, 0.2)
        v_spec_m = X_spec_m - y_spec_m
    if algorithm == 'invertB':
        #print('using ALGORITHM: INVERTB')
        y_spec_m = spec_utils.reduce_vocal_aggressively(X_spec_m, y_spec_m, 0.2)
        v_spec_m = X_spec_m - y_spec_m
    if algorithm == 'min_mag':
        #print('using ALGORITHM: MIN_MAG')
        v_spec_m = np.where(np.abs(y_spec_m) <= np.abs(X_spec_m), y_spec_m, X_spec_m)
    if algorithm == 'max_mag':
        #print('using ALGORITHM: MAX_MAG')
        v_spec_m = np.where(np.abs(y_spec_m) >= np.abs(X_spec_m), y_spec_m, X_spec_m)
    X_mag = np.abs(X_spec_m)
    y_mag = np.abs(y_spec_m)
    v_mag = np.abs(v_spec_m)

    X_image = spec_utils.spectrogram_to_image(X_mag)
    y_image = spec_utils.spectrogram_to_image(y_mag)
    v_image = spec_utils.spectrogram_to_image(v_mag)
    
    if algorithm == 'invert':
        cv2.imwrite('{}_X.png'.format(o), X_image)
        cv2.imwrite('{}_y.png'.format(o), y_image)
        cv2.imwrite('{}_v.png'.format(o), v_image)    
        
        sf.write('{}_X.wav'.format(o), spec_utils.cmb_spectrogram_to_wave(X_spec_m, mp), mp.param['sr'])
        sf.write('{}_y.wav'.format(o), spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp), mp.param['sr'])
    
    sf.write('{}.wav'.format(o), spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp), mp.param['sr'])


def whatParameterDoIUseForThisModel(modelname): # lol idk what to call this one
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

class inference:
    def __init__(self, input, param, ptm, gpu=-1, hep='none', wsize=320, agr=0.07, tta=False, oi=False, de=False, v=False, spth='separated', fn='', pp=False, arch='default'):
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
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
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
            pred = spec_utils.mask_silence(pred, pred_inv)
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
            print('done')
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), wave, mp.param['sr'])
            sf.write(os.path.join('ensembled/temp', 'tempI.wav'.format(basename, model_name, stems['inst'])), wave, mp.param['sr'])
            #vocals
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
            print('done')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), wave, mp.param['sr'])
            sf.write(os.path.join('ensembled/temp', 'tempV.wav'.format(basename, model_name, stems['vocals'])), wave, mp.param['sr'])

            if self.oi:
                with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(y_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)
                with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
                    image = spec_utils.spectrogram_to_image(v_spec_m)
                    _, bin_image = cv2.imencode('.jpg', image)
                    bin_image.tofile(f)

            print('Performing Deep Extraction...')
            #take_lowest_val(param, o, inp, algorithm='invert')
            take_lowest_val('modelparams/1band_sr44100_hl512.json',
                            'ensembled/temp/difftemp_v',
                            ['ensembled/temp/tempI.wav','ensembled/temp/tempV.wav'],
                            algorithm='min_mag')
            take_lowest_val('modelparams/1band_sr44100_hl512.json',
                            'ensembled/temp/difftemp',
                            ['ensembled/temp/tempI.wav','ensembled/temp/difftemp_v.wav'],
                            algorithm='invertB')
            os.rename('ensembled/temp/difftemp.wav',self.spth + '/{}_{}_DeepExtraction_Instruments.wav'.format(basename, model_name))
            print('Complete!')
            if isColab == False and os.path.isdir('ensembled/stage_1') == False:
                for i in range(1,13):
                    os.makedir(rf'ensembled/stage_{i}')
            clear_folder('ensembled/temp')
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
        opt = {'format': 'best', 'outtmpl': inputsha, 'updatetime': False, 'nocheckcertificate': True}
        print('Link detected')
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--isVocal','-v', action='store_true', help='Flip Instruments and Vocals output (Only for Vocal Models)') # FLIP
    p.add_argument('--output_image', '-I', action='store_true', help='Export Spectogram in an image format')
    p.add_argument('--postprocess', '-p', action='store_true', help='Masks instrumental part based on the vocals volume.')
    p.add_argument('--tta', '-t', action='store_true', help='Perform Test-Time-Augmentation to improve the separation quality.')
    p.add_argument('--suppress', '-s', action='store_true', help='Hide Warnings')

    p.add_argument('--input', '-i', required=True, help='Input')
    p.add_argument('--pretrained_model', '-P', type=str, default='', required=True, help='Pretrained model')
    p.add_argument('--nn_architecture', '-n', type=str, choices=['default', '33966KB', '123821KB', '129605KB','537238KB'], default='default', help='Model architecture')
    p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'mirroring', 'mirroring2'], default='none', help='Working with extending a low band model.')

    p.add_argument('--gpu', '-g', type=int, default=-1, help='Use GPU for faster processing')
    p.add_argument('--model_params', '-m', type=str, default='', required=True, help="Model's parameter")
    p.add_argument('--window_size', '-w', type=int, default=512, help='Window size')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.07, help='Aggressiveness of separation')

    p.add_argument('--deepextraction', '-D', action='store_true', help='Deeply remove vocals from instruments')
    p.add_argument('--model_ens', '-me', action='store_true', help='Use all 12 models and combine results') # MODEL RESULT ENSEMBLING
    p.add_argument('--convert_all', '-c', action='store_true', help='Split all tracks in tracks/ folder') # ITERATE ALL TRACKS
    p.add_argument('--useAllModel', '-a', type=str, choices=['none', 'v5', 'v5_new', 'all'], default='none', help='Use all models') # ITERATE TO MODEL

    p.add_argument('--isColab', action='store_true', help='Saves all temporary files to /content/temp')
    args = p.parse_args()
    arch = args.nn_architecture
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
    if args.suppress:
        warnings.filterwarnings("ignore")
    if os.path.isdir('ensembled/stage_1') == False:
        for i in range(1,13):
            os.mkdir('ensembled/stage_{}'.format(i))
    clear_folder('ensembled/temp')
    if args.useAllModel == 'v5':
        if args.convert_all:
            for tracks in glob.glob('tracks/*'):
                for models in glob.glob('models/v5/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    process = inference(tracks, model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                    process.inference()
        elif 'https://' in args.input:
            for models in glob.glob('models/v5/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.YouTube()
        else:
            for models in glob.glob('models/v5/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.inference()
    elif args.useAllModel == 'v5_new':
        if args.convert_all:
            for tracks in glob.glob('tracks/*'):
                for models in glob.glob('models/v5_new/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    process = inference(tracks, model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                    process.inference()
        elif 'https://' in args.input:
            for models in glob.glob('models/v5_new/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.YouTube()
        else:
            for models in glob.glob('models/v5_new/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.inference()

    #why would you do this... rip google drive lol
    elif args.useAllModel == 'all':
        if args.convert_all:
            for tracks in glob.glob('tracks/*'):
                for models in glob.glob('models/v5/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    process = inference(tracks, model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                    process.inference()
                for models in glob.glob('models/v5_new/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    print('With {}:'.format(models))
                    inference(tracks, model_params, models, args.gpu, args.high_end_process, args.window_size, args.aggressiveness, args.tta, args.output_image, args.deepextraction, args.postprocess,arch=arch)
        elif 'https://' in args.input:
            for models in glob.glob('models/v5/*'):
                for tracks in glob.glob('tracks/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    process = inference(tracks,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                    process.YouTube()
            for models in glob.glob('models/v5_new/*'):
                for tracks in glob.glob('tracks/*'):
                    model_params = whatParameterDoIUseForThisModel(models)
                    arch = whatArchitectureIsThisModel(models)
                    process = inference(tracks,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                    process.YouTube()
        else:
            for models in glob.glob('models/v5/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.inference()
            for models in glob.glob('models/v5_new/*'):
                model_params = whatParameterDoIUseForThisModel(models)
                arch = whatArchitectureIsThisModel(models)
                process = inference(args.input,model_params,models,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch)
                process.inference()
    elif args.convert_all:
        for tracks in glob.glob('tracks/*'):
            print('Now splitting: {}'.format(os.path.splitext(os.path.basename(tracks))[0]))
            process = inference(tracks,args.model_params,args.pretrained_model,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch,v=args.isVocal)
            process.inference()
            print('---------------------------------------------------------')
    elif args.model_ens:
        if args.isColab:
            print('Temporary files will be saved in /content/temp/*')
            if os.path.isdir('/content/temp'):
                pass
            else:
                for stages in range(1,10+1):
                    os.makedirs(rf'/content/temp/stage_{stages}')
                os.makedirs(r'/content/temp/temp')

            temp_ens_path = '/content/temp/temp'
        else:
            temp_ens_path = 'ensembled/temp'
        loc = ['models/v5_new/','models/v5/']
        ensmodels = [f'{loc[0]}HighPrecison_4band_arch-124m_1.pth',
                    f'{loc[0]}HighPrecison_4band_arch-124m_2.pth',
                    f'{loc[0]}LOFI_2band-1_arch-34m.pth',
                    f'{loc[0]}LOFI_2band-2_arch-34m.pth',
                    f'{loc[0]}NewLayer_4band_arch-130m_1.pth',
                    f'{loc[0]}NewLayer_4band_arch-130m_2.pth',
                    f'{loc[1]}MGM-v5-2Band-32000-_arch-default-BETA1.pth',
                    f'{loc[1]}MGM-v5-2Band-32000-_arch-default-BETA2.pth',
                    f'{loc[1]}MGM-v5-4Band-44100-_arch-default-BETA1.pth',
                    f'{loc[1]}MGM-v5-4Band-44100-_arch-default-BETA2.pth',
                    f'{loc[1]}MGM-v5-MIDSIDE-44100-_arch-default-BETA1.pth',
                    f'{loc[1]}MGM-v5-MIDSIDE-44100-_arch-default-BETA2.pth']
        r = 0
        #print('Splitting {}...'.format(os.path.splitext(os.path.basename(args.input))[0]))

        for models in ensmodels:
            print('USING MODEL: {}'.format(models))
            #break
            model_params = whatParameterDoIUseForThisModel(models)
            arch = whatArchitectureIsThisModel(models) # import as nets
            process = inference(args.input,
                                model_params,
                                models,
                                gpu=args.gpu,
                                hep=args.high_end_process,
                                wsize=args.window_size,
                                agr=args.aggressiveness,
                                tta=args.tta,
                                oi=args.output_image,
                                de=args.deepextraction,
                                pp=args.postprocess,
                                fn=r,
                                spth=temp_ens_path,
                                arch=arch)
            process.inference()
            r += 1
            print('---------------------------------------------------------')
        if args.isColab:
            temp_ens_path = '/content/temp/'
            savedtemp = '/content/temp'
        else:
            temp_ens_path = 'ensembled'
            savedtemp = 'ensembled'
        basename_list = []
        for a in enumerate(ensmodels):
            b = os.path.splitext(os.path.basename(a[1]))[0]
            basename_list.append(str(a[0])+'_'+b)
        
        param = 'modelparams/ensemble.json'
        #param = 'modelparams/1band_sr44100_hl512.json'
        print('Ensembling Instrumental...')
        progress_bar = tqdm(total=55)
        for a,b in zip(basename_list,range(len(basename_list))): # stage 1
            
            if b*2 == 0:
                c = 1
            else:
                c += 1 # temporary b+1, +1 index offset
                if c >= len(basename_list)-1:
                    break

            take_lowest_val(param,f'{temp_ens_path}/stage_1/'+str(b),[savedtemp+'/temp/'+a+'_Instruments.wav',savedtemp+'/temp/'+basename_list[c]+'_Instruments.wav'],algorithm='min_mag',supress=True)
            progress_bar.update(1)
        # nested loop
        for folder in range(2,12+1):
            for a in range(len(basename_list)-folder):
                if a*2 == 0:
                    c = 1 
                else:
                    c += 1
                    if c >= len(basename_list)-folder:
                        break
                #temp_ens_path+f'/stage_{folder-1}/{a}.wav
                if os.path.isfile(temp_ens_path+f'/stage_{folder-1}/{a+1}.wav') == False:
                    break
                take_lowest_val(param,temp_ens_path+f'/stage_{folder}/'+str(a),[temp_ens_path+f'/stage_{folder-1}/{a}.wav',temp_ens_path+f'/stage_{folder-1}/{c}.wav'],algorithm='min_mag',supress=True)
                progress_bar.update(1)
        progress_bar.close()
        bsnme = os.path.splitext(os.path.basename(args.input))[0]
        if True:
            final_ens = os.path.join(f'{temp_ens_path}/stage_10','0.wav')
            if os.path.isfile('separated/{}_Ensembled_Instruments.wav'.format(bsnme)):
                rename = os.path.join(f'{temp_ens_path}/stage_10',bsnme + '{}_Ensembled_Instruments.wav'.format(random.randint(0,1000)))
            else:
                rename = os.path.join(f'{temp_ens_path}/stage_10',bsnme + '_Ensembled_Instruments.wav')
            os.rename(final_ens,rename)
            final_ens = '{}/stage_10/{}_Ensembled_Instruments.wav'.format(temp_ens_path,bsnme)
            if os.path.isfile('separated/' + bsnme):
                final_ens = '{}/stage_10/{}_{}_Ensembled_Instruments.wav'.format(temp_ens_path,random.randint(0,22),bsnme)
            shutil.move(final_ens,'separated/')

        print('Ensembling Vocals...')
        progress_bar = tqdm(total=55)
        for a,b in zip(basename_list,range(len(basename_list))): # stage 1
            
            if b*2 == 0:
                c = 1
            else:
                c += 1 # temporary b+1, +1 index offset
                if c >= len(basename_list)-1:
                    break
            take_lowest_val(param,f'{temp_ens_path}/stage_1/'+str(b),[savedtemp+'/temp/'+a+'_Vocals.wav',savedtemp+'/temp/'+basename_list[c]+'_Vocals.wav'],algorithm='max_mag',supress=True)
            progress_bar.update(1)
        # nested loop
        for folder in range(2,12+1):
            for a in range(len(basename_list)-folder):
                if a*2 == 0:
                    c = 1 
                else:
                    c += 1
                    if c >= len(basename_list)-folder:
                        break
                #temp_ens_path+f'/stage_{folder-1}/{a}.wav
                if os.path.isfile(temp_ens_path+f'/stage_{folder-1}/{a+1}.wav') == False:
                    break
                take_lowest_val(param,temp_ens_path+f'/stage_{folder}/'+str(a),[temp_ens_path+f'/stage_{folder-1}/{a}.wav',temp_ens_path+f'/stage_{folder-1}/{c}.wav'],algorithm='max_mag',supress=True)
                progress_bar.update(1)
        progress_bar.close()
        if True:
            final_ens = os.path.join(f'{temp_ens_path}/stage_10','0.wav')
            if os.path.isfile('separated/{}_Ensembled_Vocals.wav'.format(bsnme)):
                rename = os.path.join(f'{temp_ens_path}/stage_10',bsnme + '{}_Ensembled_Vocals.wav'.format(random.randint(0,1000)))
            else:
                rename = os.path.join(f'{temp_ens_path}/stage_10',bsnme + '_Ensembled_Vocals.wav')
            os.rename(final_ens,rename)
            final_ens = '{}/stage_10/{}_Ensembled_Vocals.wav'.format(temp_ens_path,bsnme)
            if os.path.isfile('separated/' + bsnme):
                final_ens = '{}/stage_10/{}_{}_Ensembled_Vocals.wav'.format(temp_ens_path,random.randint(0,22),bsnme)
            shutil.move(final_ens,'separated/')
        for i in range(1,10+1):
            clear_folder('{}/stage_{}/'.format(temp_ens_path,i))
        if args.isColab:
            temp_ens_path = '/content/temp'
        clear_folder(f'{temp_ens_path}/temp')
    else:
        if 'https://' in args.input:
            warnings.filterwarnings("ignore")
            process = inference(args.input,args.model_params,args.pretrained_model,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch,v=args.isVocal)
            process.YouTube()
        else: # single
            process = inference(args.input,args.model_params,args.pretrained_model,gpu=args.gpu,hep=args.high_end_process,wsize=args.window_size,agr=args.aggressiveness,tta=args.tta,oi=args.output_image,de=args.deepextraction,pp=args.postprocess,arch=arch,v=args.isVocal)
            process.inference()
            #### old inference(args.input, args.model_params, args.pretrained_model, args.gpu, args.high_end_process, args.window_size, args.aggressiveness, tta=args.tta, oi=args.output_image, de=args.deepextraction, pp=args.postprocess, v=args.isVocal)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
