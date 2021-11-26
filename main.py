from pathvalidate import sanitize_filename
import soundfile as sf
import numpy as np
import youtube_dl
import importlib
import argparse
import warnings
import os.path
import librosa
import hashlib
import types
import shutil

import torch

import time
import glob
import cv2
import sys
import os

from lib.model_param_init import ModelParameters
from lib import vr as _inference
from lib import automation
from lib import spec_utils


class hide_opt:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class inference:
    def __init__(self, _input, param, ptm, gpu=-1, hep='none', wsize=320, agr=0.07, tta=False, oi=False, de=False, v=False, spth='separated', fn='', pp=False, arch='default',
                pp_thres = 0.2, mrange = 32, fsize = 64):
        self.input = _input
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
        def loadModel():
            global mp, device, model
            try:
                print('loading model...', end=' ')
                mp = ModelParameters(self.param)
                device = torch.device('cpu')
                model = nets.CascadedASPPNet(mp.param['bins'] * 2)
                model.load_state_dict(torch.load(self.ptm, map_location=device))
                if torch.cuda.is_available() and self.gpu >= 0:
                    device = torch.device('cuda:{}'.format(self.gpu))
                    model.to(device)
            except Exception as e:
                return str(e)
            return True
        load_counter = 0
        while True:
            load_counter += 1
            if load_counter == 5:
                quit('An error has occurred: {}'.format(a))
            a = loadModel()
            if not type(a) == bool:
                print('Model loading failed, trying again...')
            else:
                del a
                break
        print('done')
        # stft of wave source -------------------------------
        print('stft of wave source...', end=' ')
        if self.fn != '':
            basename = self.fn
        else:
            basename = os.path.splitext(os.path.basename(self.input))[0]
        X_spec_m, input_high_end_h, input_high_end = spec_utils.loadWave(self.input, mp, hep=self.hep)
        print('done')
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
        # deep ext stems!
        stems['di'] = 'DeepExtraction_Instruments'
        stems['dv'] = 'DeepExtraction_Vocals'
        y_spec_m = pred * X_phase # instruments
        v_spec_m = X_spec_m - y_spec_m # vocals

        #Instrumental wave upscale
        if self.hep == 'bypass':
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)
        elif self.hep.startswith('mirroring'):       
            input_high_end_ = spec_utils.mirroring(self.hep, y_spec_m, input_high_end, mp)
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)  
        else:
            y_wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
        
        v_wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        #saving files------------------------
        if self.de: # deep extraction
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), y_wave, mp.param['sr'])
            print('done')
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), v_wave, mp.param['sr'])
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
            if os.path.isdir('/content/tempde') == False:
                os.mkdir('/content/tempde')

            spec_utils.spec_effects(ModelParameters('modelparams/1band_sr44100_hl512.json'),
                                    [os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Vocals')),
                                     os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Instruments'))],
                                    '/content/tempde/difftemp_v',
                                    algorithm='min_mag')
            spec_utils.spec_effects(ModelParameters('modelparams/1band_sr44100_hl512.json'),
                                    ['/content/tempde/difftemp_v.wav',
                                     os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Instruments'))],
                                    '/content/tempde/difftemp',
                                    algorithm='invert')
            os.rename('/content/tempde/difftemp.wav','/content/tempde/{}_{}_{}.wav'.format(basename, model_name, stems['di']))
            
            if os.path.isfile(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['di'])):
                os.remove(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['di']))
            shutil.move('/content/tempde/{}_{}_{}.wav'.format(basename, model_name, stems['di']),self.spth)
            # VOCALS REMNANTS
            
            if os.path.isfile(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['dv'])):
                os.remove(self.spth+'/{}_{}_{}.wav'.format(basename, model_name, stems['dv']))
            excess,_ = librosa.load('/content/tempde/difftemp_v.wav',mono=False,sr=44100)
            _vocal,_ = librosa.load(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, 'Vocals')),
                                    mono=False,sr=44100)
            # this isn't required, but just in case.
            excess, _vocal = spec_utils.align_wave_head_and_tail(excess,_vocal)
            sf.write(self.spth + '/{}_{}_{}.wav'.format(basename,model_name, stems['dv']),excess.T+_vocal.T,44100)
            print('Complete!')
        else: # args
            print('inverse stft of {}...'.format(stems['inst']), end=' ')
            model_name = os.path.splitext(os.path.basename(self.ptm))[0]
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['inst'])), y_wave, mp.param['sr'])
            print('done')
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            print('done')
            sf.write(os.path.join(self.spth, '{}_{}_{}.wav'.format(basename, model_name, stems['vocals'])), v_wave, mp.param['sr'])
            if self.oi:
                    with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
                        image = spec_utils.spectrogram_to_image(y_spec_m)
                        _, bin_image = cv2.imencode('.jpg', image)
                        bin_image.tofile(f)
                    with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
                        image = spec_utils.spectrogram_to_image(v_spec_m)
                        _, bin_image = cv2.imencode('.jpg', image)
                        bin_image.tofile(f)
        #torch.cuda.empty_cache() # clear ram <<<
    
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
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        if self.de:
            if self.v:
                stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
            os.rename("separated/{}_DeepExtraction_{}.wav".format(inmodname,stems['inst']), 'separated/' + '{}_{}_DeepExtraction_{}.wav'.format(sanitize_filename(titlename),modname,stems['inst']))
            os.rename("separated/{}_DeepExtraction_{}.wav".format(inmodname,stems['vocals']), 'separated/' + '{}_{}_DeepExtraction_{}.wav'.format(sanitize_filename(titlename),modname,stems['vocals']))
            os.remove(inputsha)
        if os.path.isfile(inputsha):
            os.remove(inputsha)



def multi_file(models, inputs):
    print('Multiple files detected.')
    print('---------------------------------------------------------------')
    for model in models:
        for track in inputs:
            print('Now processing: {} \nwith {}'.format(os.path.splitext(os.path.basename(track))[0],os.path.basename(model)))
            model_params = automation.whatParameterDoIUseForThisModel(model)
            arch = automation.whatArchitectureIsThisModel(model)
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
        if '://' in args.input:
            warnings.filterwarnings("ignore")
            process.YouTube()
        else: # single
            process.inference()


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    print('Total time: {0:.{1}f}s'.format(time.perf_counter() - start_time, 1))
