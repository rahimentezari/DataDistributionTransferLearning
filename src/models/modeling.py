import torch

import pickle
import sys
sys.path.insert(1, 'src')
from models import utils
import clip as clip
import json
from pathlib import Path
from .model import CLIP
from .simclr import models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from PIL import Image


def _convert_to_rgb(image):
    return image.convert('RGB')
def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()
        print("model", args.model)
        if 'SimCLR' in args.upstream_loss:
            self.train_preprocess = _transform(224, is_train=True)
            self.val_preprocess = _transform(224, is_train=False)
            model = getattr(models, 'SIMCLR_RN50')(ssl_mlp_dim=4096, ssl_emb_dim=256)
            state_dict = torch.load(args.model, map_location="cpu")
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model.load_state_dict(sd)
            self.model = model

        else:
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                args.model, args.device, jit=False)


            # #### from scratch
            # model_config_file = Path(__file__).parent / f"model_configs/RN50.json"
            # print('Loading model from', model_config_file)
            # with open(model_config_file, 'r') as f:
            #     model_info = json.load(f)
            # self.model = CLIP(**model_info)


            # self.model, self.train_preprocess, self.val_preprocess = clip.load(  ### loads openai checkpoint
            #     'RN50', args.device, jit=False)
            # print(self.model)
        
        self.cache_dir = './'

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        # print("weights", weights.shape)
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        # print(inputs.shape)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

    def class_head(self):
        return self.classification_head




