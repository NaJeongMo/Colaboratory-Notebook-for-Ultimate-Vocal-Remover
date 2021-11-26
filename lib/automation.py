def whatParameterDoIUseForThisModel(modelname):
    modelname = modelname.lower()
    if '4band' in modelname:
        if 'v2' in modelname:
            parameter = 'modelparams/4band_v2.json'
        else:
            parameter = 'modelparams/4band_44100.json'
    elif '3band' in modelname:
        if 'msb2' in modelname:
            parameter = 'modelparams/3band_44100_msb2.json'
        else:
            parameter = 'modelparams/3band_44100.json'
    elif 'midside' in modelname:
        parameter = 'modelparams/3band_44100_mid.json'
    elif '2band' in modelname:
        if '32000' in modelname:
            parameter = 'modelparams/2band_32000.json' 
        else:
            parameter = 'modelparams/2band_48000.json'
    elif 'lofi' in modelname:
        parameter = 'modelparams/2band_44100_lofi.json'
    else:
        if '32000' in modelname:
            parameter = 'modelparams/1band_sr32000_hl512.json'
        else:
            parameter = 'modelparams/1band_sr44100_hl512.json'
    return parameter

def whatArchitectureIsThisModel(modelname):
        modelname = modelname.lower()
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
