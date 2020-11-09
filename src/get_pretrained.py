import gdown

from .config_predict import ConfigPredict


def get_pretrained(quiet=False):
    gdown.download(
        ConfigPredict.pretrained_weights_url,
        ConfigPredict.pretrained_weights_path,
        quiet,
    )
