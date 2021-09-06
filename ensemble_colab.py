import soundfile as sf
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os.path
import librosa
import random
import shutil

import glob
import sys
import os

from lib.model_param_init import ModelParameters
import main
from main import inference
from lib import automation
from lib import spec_utils


def ensembleIteration(count):
    count -= 1
    count_temp = count
    result = 0
    for i in range(count):
        i = count_temp - i
        result += i
    return result


def ensemble(input,model=[],algorithms=['min_mag','max_mag']):
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
    # filename defining
    ins_algo = algorithms[0]
    voc_algo = algorithms[1]
    basename = os.path.splitext(os.path.basename(input))[0]
    print('Processing {} with {} models...'.format(os.path.splitext(os.path.basename(input))[0],len(model)))
    r = 0
    print('---------------------------------------------------------')
    for models in model:
        print('USING MODEL: {}'.format(os.path.splitext(os.path.basename(models))[0]))
        model_params = automation.whatParameterDoIUseForThisModel(models)
        arch = automation.whatArchitectureIsThisModel(models)
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

    stem_spth = {'inst':f'separated/{basename}_Ensembled_Instruments.wav',
                'vocals':f'separated/{basename}_Ensembled_Vocals.wav'}

    bsnme = os.path.splitext(os.path.basename(input))[0]

    params = ModelParameters('modelparams/1band_sr44100_hl512.json')
    for a in range(0,r):
        if a + 1 > r-1:
            break
        
        spec_utils.spec_effects(params,
                                [f'{savedtemp}/temp/{a}_{modelname[a]}_Instruments.wav',f'{savedtemp}/temp/{a+1}_{modelname[a+1]}_Instruments.wav'],
                                f'{savedtemp}/stage_1/{a}',
                                algorithm=ins_algo)
        progress_bar.update(1)#
    # set 0
    for folder in range(2,r):
        for a in range(r-folder):
            spec_utils.spec_effects(params,
                                    [f'{savedtemp}/stage_{folder-1}/{a}.wav',f'{savedtemp}/stage_{folder-1}/{a+1}.wav'],
                                    f'{savedtemp}/stage_{folder}/{a}',
                                    algorithm=ins_algo)
            progress_bar.update(1)
    file = stem_spth['inst']
    if os.path.isfile(file):
        os.remove(file)
    final_ens = os.path.join(f'{savedtemp}/stage_{r-1}/','0.wav')
    os.rename(final_ens,
                os.path.join(f'{savedtemp}/stage_{r-1}/',f'{bsnme}_Ensembled_Instruments.wav'))
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
        #spec_effects(mp, inp, o, algorithm='invert')
        spec_utils.spec_effects(params,
                                [f'{savedtemp}/temp/{a}_{modelname[a]}_Vocals.wav',f'{savedtemp}/temp/{a+1}_{modelname[a+1]}_Vocals.wav'],
                                f'{savedtemp}/stage_1/{a}',
                                algorithm=voc_algo)
        progress_bar.update(1)
    # set 0
    for folder in range(2,r):
        for a in range(r-folder):
            spec_utils.spec_effects(params,
                                    [f'{savedtemp}/stage_{folder-1}/{a}.wav',f'{savedtemp}/stage_{folder-1}/{a+1}.wav'],
                                    f'{savedtemp}/stage_{folder}/{a}',
                                    algorithm=voc_algo)
            progress_bar.update(1)
    file = stem_spth['vocals']
    if os.path.isfile(file):
        os.remove(file)
    final_ens = os.path.join(f'{savedtemp}/stage_{r-1}/','0.wav')
    os.rename(final_ens,
                os.path.join(f'{savedtemp}/stage_{r-1}/',f'{bsnme}_Ensembled_Vocals.wav'))
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
p.add_argument('--algo','-a', nargs='+', default=['min_mag','max_mag'], help='Algorithm to be used for instrumental and acapella. (In order)')

p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'mirroring', 'mirroring2'], default='none')
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
ensemble(args.input,args.model_ens,algorithms=args.algo)
