import numpy as np
import PIL
import requests
import torch
from icevision import ClassMap, Dataset, faster_rcnn, tfms

from .config_predict import ConfigPredict
from .get_pretrained import get_pretrained


class Predicter:
    """A predicter class to streamline all the necessary steps"""

    def __init__(self, download=True):
        """set download to False if already downloaded the model"""
        self.img = None
        # image transforms
        self.tfms = tfms.A.Adapter([tfms.A.Normalize()])
        # class map : (e.g. Person / Car)
        self.class_map = ClassMap(ConfigPredict.class_map)
        # model contaner
        self.model_container = globals().get(ConfigPredict.use_model)
        # cpu for predict
        self.device = torch.device("cpu")
        # model
        self.model = self.get_model(download)

    def get_model(self, download=True):
        """return pretrained model as per config"""
        if download:
            get_pretrained()
        return self.load_model()

    def load_model(self):
        """separate out model loading, and call this in self.get_model
        after getting pretrained model downloaded"""
        if ConfigPredict.use_model == "faster_rcnn":
            model = self.model_container.model(
                backbone=None, num_classes=len(self.class_map), pretrained=False
            )
            model.load_state_dict(
                torch.load(
                    ConfigPredict.pretrained_weights_path, map_location=self.device
                )
            )
        return model

    def image_from_url(self, url):
        """get a new image to predict on"""
        res = requests.get(url, stream=True)
        img = PIL.Image.open(res.raw)
        self.img = np.array(img)

    def predict(self):
        """run the inference on valid_ds - and give out the predictions"""
        ds = Dataset.from_images([self.img], self.tfms)
        batch, samples = self.model_container.build_infer_batch(ds)
        preds = self.model_container.predict(self.model, batch=batch)
        img, pred = samples[0]["img"], preds[0]
        return img, pred
