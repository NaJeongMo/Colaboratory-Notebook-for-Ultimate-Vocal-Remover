# Colaboratory Notebook for Ultimate Vocal Remover
<br>Trained models provided in this notebook are by [Anjok07](https://github.com/Anjok07), and [aufr33](https://github.com/aufr33).</br>
<br>OFFICIAL UVR GITHUB PAGE: [here](https://github.com/Anjok07/ultimatevocalremovergui).</br>
<br>OFFICIAL CLI Version: [here](https://github.com/tsurumeso/vocal-remover).</br>
<sup><br>Powered by [tsurumeso](http://github.com/tsurumeso/).
<br>Colaboratory version by [AudioHacker](https://www.youtube.com/channel/UC0NiSV1jLMH-9E09wiDVFYw), **Hv#3868**.</br></sup>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

### Training Similarity extractor model
- You'll need any tools that can filter mid and side
  - Download original CLI version [here](https://github.com/tsurumeso/vocal-remover/archive/refs/heads/master.zip)
  - ```python3 train.py --dataset "path/to/dataset/" --sr 44100 --hop_length 512 --n_fft 2048```
```
path/to/dataset/
  +- instruments/
  |    +- 01_foo_mid.wav ( this is dual mono )
  |    +- 02_bar_mid.mp3 ( this is dual mono )
  |    +- ...
  +- mixtures/
       +- 01_foo_mixture.wav
       +- 02_bar_mixture.mp3
       +- ...
```
