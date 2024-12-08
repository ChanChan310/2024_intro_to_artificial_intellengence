import torch
import torch.nn as nn
import hparams as hp
import utils

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from modules import LengthRegulator, CBHG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels)
        self.postnet = CBHG(hp.num_mels, K=8,
                            projections=[256, hp.num_mels])
        self.last_linear = Linear(hp.num_mels * 2, hp.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        # Debugging input sizes
        # print(f"src_seq: max={src_seq.max()}, min={src_seq.min()}, shape={src_seq.shape}")
        # print(f"src_pos: max={src_pos.max()}, min={src_pos.min()}, shape={src_pos.shape}")

        # Ensure indices are valid for embedding layers
        assert src_seq.max() < self.encoder.src_word_emb.num_embeddings, "src_seq contains invalid indices."
        assert src_pos.max() < self.encoder.position_enc.num_embeddings, "src_pos contains invalid indices."

        # Encoder output
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output, target=length_target, alpha=alpha, mel_max_length=mel_max_length
            )
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output, mel_pos, mel_max_length)

            return mel_output, mel_postnet_output, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output, alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output



if __name__ == "__main__":
    # Test
    model = FastSpeech()
    print(sum(param.numel() for param in model.parameters()))
