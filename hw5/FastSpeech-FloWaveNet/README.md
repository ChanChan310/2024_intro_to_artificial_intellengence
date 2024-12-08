# FastSpeech-FloWaveNet
This repository is adapted from [FastSpeech-Pytorch](https://github.com/xcmyz/FastSpeech) from `xcmyz`, [FloWaveNet](https://github.com/ksw0306/FloWaveNet) from `kws0306`, and [FastSpeech-FloWaveNet](https://github.com/cjlin8787/FastSpeech-FloWaveNet) from `cjlin8787`. Instead of using WaveGlow, FastSpeech from this repository use FloWaveNet as the vocoder. Many thanks to the original authors. Please check their repositories for more detailed description.

## Prepare Pretrained Model
1. Download pretrained model of flowavenet from [here](https://drive.google.com/drive/folders/1AqdZaqAFRcBns4UDveLj3s4o4jOwIos8?usp=drive_link), and put it under `flowavenet/pretrained_model`.
2. Download pretrained model of fastspeech from [here](https://drive.google.com/file/d/1vMrKtbjPj9u_o3Y-8prE6hHCc6Yj4Nqk/view?usp=sharing), and put it under `new_model`.

## Setup Environment
Using [colab](https://colab.research.google.com/drive/1wSFNqDYhSQ98oIo4viGenG3G1nVK24d3?usp=sharing)

On local:
Please check the [Pytorch](https://pytorch.org) website if CUDA version needs to be downloaded.
```
conda create -n fastspeech python=3.8 --yes
conda config --add channels conda-forge
conda activate fastspeech
conda install conda-forge::mamba --yes
mamba install --file requirements.txt -c pytorch -c defaults -c anaconda -c conda-forge --yes
```
If you encounter errors, try install the packages seperately by using:
```
mamba install "package_name<version"
```
Or create a new env with python=3.9 and try again.

## Synthesize
The result will be written into `results/${step}_${flowavenets_step}`. Please change `flowavenet_step` according to the file downloaded from [Prepare FloWaveNet Model](#prepare-flowavenet-model) to synthesize audio in different quality.
```
python synthesize.py --file generate.txt --step 135000 --flowavenet_step 126764
```