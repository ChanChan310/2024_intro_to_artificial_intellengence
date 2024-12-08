import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import audio
import dataset
import hparams as hp
import model as M
import text
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(
        torch.load(os.path.join(hp.checkpoint_path, checkpoint_path), map_location=device)['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    if src_pos.max() >= model.module.encoder.position_enc.num_embeddings:
        raise ValueError(
            f"Input sequence length ({src_pos.max()}) exceeds positional encoding limit "
            f"({model.module.encoder.position_enc.num_embeddings}). Reduce input length or increase model capacity."
        )
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).to(device).long()
    src_pos = torch.from_numpy(src_pos).to(device).long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(file):
    data_list = list()
    for t in open(file, 'r').readlines():
        data_list.append(text.text_to_sequence(t, hp.text_cleaners))

    return data_list


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--flowavenet_step', type=int, default=126764)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--file", type=str, default='')
    args = parser.parse_args()

    print("use griffin-lim and flowavenet")
    flowavenet = utils.get_FloWaveNet(args.flowavenet_step, device)
    model = get_DNN(args.step)
    data_list = get_data(args.file)

    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, args.alpha)
        outdir = os.path.realpath(f'results/{args.step}_{args.flowavenet_step}')
        os.makedirs(outdir, exist_ok=True)
        mel_path = os.path.join(outdir, f'{i}_mel.npy')
        wav_path = os.path.join(outdir, f'{i}_inverse_audio.wav')
        np.save(mel_path, mel.numpy().astype(np.float32))
        audio.tools.inv_mel_spec(mel, wav_path)
        flowavenet.inference(
            wav_path, mel_path, os.path.join(outdir, f'{i}_flowavenet_audio.wav')
        )
        print("Done", i + 1)
'''
    s_t = time.perf_counter()
    for i in range(100):
        for _, phn in enumerate(data_list):
            _, _, = synthesis(model, phn, args.alpha)
        print(i)
    e_t = time.perf_counter()
    print((e_t - s_t) / 100.)
'''
