# from icevision.all import *
# from fastai.callback.wandb import *
# from fastai.callback.tracker import SaveModelCallback
import gc

import torch
from icevision import (ClassMap, COCOMetric, COCOMetricType, Dataset,
                       faster_rcnn, tfms)

from .config import Config
from .parsing import do_parsing


class Trainer:
    """A trainer class to hold all the relevant stuff:
    train/valid records, train/valid data transforms,
    model, fastai's learn object, metric etc.
    """

    def __init__(self):
        # parse records for fixing
        self.train_records, self.valid_records = do_parsing()

        # image transforms
        self.train_tfms = tfms.A.Adapter(
            [
                *tfms.A.aug_tfms(size=Config.img_size, presize=Config.img_presize),
                tfms.A.Normalize(),
            ]
        )
        # self.valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(Config.img_size), tfms.A.Normalize()])
        self.valid_tfms = tfms.A.Adapter(
            [
                *tfms.A.aug_tfms(
                    size=Config.img_size,
                    presize=Config.img_presize,
                    shift_scale_rotate=None,
                ),
                tfms.A.Normalize(),
            ]
        )

        # Datasets
        self.train_ds = Dataset(self.train_records, self.train_tfms)
        self.valid_ds = Dataset(self.valid_records, self.valid_tfms)

        # class map : Person / Car
        self.class_map = ClassMap(Config.class_map)

        # model contaner
        self.model_container = globals().get(Config.use_model)

        # Dataloaders
        self.train_dl = self.model_container.train_dl(
            self.train_ds,
            batch_size=Config.img_bs,
            num_workers=Config.num_workers,
            shuffle=True,
        )
        self.valid_dl = self.model_container.valid_dl(
            self.valid_ds,
            batch_size=Config.img_bs,
            num_workers=Config.num_workers,
            shuffle=False,
        )

        # gpu / cpu
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # model and metric
        self.model = self.get_model()
        self.metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

        self.learn = self.get_learn()

        # to be updated once training completes
        #  call self.infer_valid()
        self.infer_dl = None

    def get_model(self):
        """return model faster rcnn / efficientdet as per choice"""
        if Config.use_model == "faster_rcnn":
            model = self.model_container.model(
                backbone=None, num_classes=len(self.class_map)
            )
        elif Config.use_model == "efficientdet":
            _model_name = "efficientdet_d1"
            model = self.model_container.model(
                model_name=_model_name,
                num_classes=len(self.class_map),
                img_size=Config.img_size,
            )
        model = model.to(self.device)
        return model

    def get_learn(self):
        """get the fastai learn object"""
        return self.model_container.fastai.learner(
            dls=[self.train_dl, self.valid_dl], model=self.model, metrics=self.metrics
        )

    def release_cuda(self):
        """garbage collect, and release memory in case of CUDA out of memory"""
        del self.learn, self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.get_model()
        self.learn = self.get_learn()

    def fine_tune(self):
        """finetune the model - first unfreeze only the last layers, before all of them"""
        self.learn.fine_tune(
            Config.num_postfreeze_epochs, 2e-4, freeze_epochs=Config.num_freeze_epochs
        )

    def infer_valid(self):
        """run the inference on valid_ds - and give out the predictions"""
        self.infer_dl = self.model_container.infer_dl(
            self.valid_ds, batch_size=Config.img_bs
        )
        samples, preds = self.model_container.predict_dl(self.model, self.infer_dl)
        return samples, preds

    def save_model(self):
        """save the model in path set in config.py's Config class"""
        if Config.do_savemodel:
            torch.save(self.model.state_dict(), Config.model_savepath)
        else:
            print("`Config.do_savemodel` is set as False")
