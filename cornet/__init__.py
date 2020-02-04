import torch
import torch.utils.model_zoo

from cornet.cornet_z import CORnet_Z
from cornet.cornet_z import HASH as HASH_Z
from cornet.cornet_r import CORnet_R
from cornet.cornet_r import HASH as HASH_R
from cornet.cornet_rt import CORnet_RT
from cornet.cornet_rt import HASH as HASH_RT
from cornet.cornet_s import CORnet_S
from cornet.cornet_s import HASH as HASH_S


def get_model(model_letter, pretrained=False, map_location=None, **kwargs):
    model_letter = model_letter.upper()
    model_hash = globals()[f'HASH_{model_letter}']
    model = globals()[f'CORnet_{model_letter}'](**kwargs)
    model = torch.nn.DataParallel(model)
    if pretrained:
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data['state_dict'])
    return model


def cornet_z(pretrained=False, map_location=None):
    return get_model('z', pretrained=pretrained, map_location=map_location)


def cornet_r(pretrained=False, map_location=None, times=5):
    return get_model('r', pretrained=pretrained, map_location=map_location, times=times)


def cornet_rt(pretrained=False, map_location=None, times=5):
    return get_model('rt', pretrained=pretrained, map_location=map_location, times=times)


def cornet_s(pretrained=False, map_location=None):
    return get_model('s', pretrained=pretrained, map_location=map_location)