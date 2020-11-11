### Library and Model Choices

- The library chosen for this project is [icevision](https://github.com/airctic/icevision)
    - there are multiple other valid oprions like detectron2 or stand alone efficientdet libraries
    - but, `icevision` has a very nice scaffolding in terms displaying the data after augmentation, and outputs as they match up after prediction
    - this gives very nice (and quick intuition) about what data the model is going to see, and how well it's doing
    - also, since it's based on `fastai` it has taken in the niceties from there - like model freezing and unfreezing - one cycle training etc.

    having these out of the box was very helpful

- In terms of model - `icevision` supports two of the most common ones - `faster-rcnn` and `efficientdet` (for which it depends on rwightman's efficientdet library)
    - in my project, after experimentation - I went ahead with `faster-rcnn` with `resnet50` backbone
    - it seemed to train much better than `efficientdet` - which seemed to face more *catastrophic forgetting* - so, even though they would ultimately reach similar `mAP` **(~0.29)**, `efficientdet` was a little behind, and took longer
    - but, this is an avenue that needs to be explored further - as `efficientdet` was almost *half* the size of the `faster-rcnn` model - which would be an important consideration for deployemnt
    - **further ideas to explore**:
        trying out `DETR` which seemingly has a very interesting architecture

### Platform Choices
- for training the model, colab GPU was used on `code-server` using my [forked version of `colabcode`](https://github.com/partham16/colabcode) (adds a bit more niceties like `ohmybash`, `powerline` bash prompt etc.)
- for testing the model, a new virtual environment was created on `python 3.8` on *visual studio codespaces* - to see if it's working from scratch
- furthermore, `streamlit` and `docker` was used to package the resulting model for serving - for which an [`azure container instance`](http://evobjdetpc.eastus.azurecontainer.io:8501/) was used
    - *yet to be merged with `master` branch*
- finally, a `github action` was set for *CI/CD* - that runs the committed code on `python 3.8`

### Data and Augmentation choices
- a few cases had a bounding box area of `0` - which were to be discarded
- also, at the adjunction of `icevision` and `albumentations`, certain images were having [very low bounding box width compared to the image width](https://github.com/airctic/icevision/issues/467) for albumentations to process them properly - that's why `42` images were discarded!
    - note: the *issue* mentioned above asserts that it happens for `efficientdet`, not `faster-rcnn` - but, it happened for both the models, and presumably is due to albumentations (needs to be pinned down, properly) - not a particular type of model
- finally, the images were split into `75:25` for training and validation
- the training images were augmented by sizing them into `384` pixels, as well as `shift_scale_roate`, `brightness`, `blur`, `rgb_shift` etc.
    - an example of data augmentation can be seen in the [accompanying colab notebook](https://github.com/partham16/ev_objdet_pc/blob/master/ev_train.ipynb)
- for validation images, only the `brightness`, `blur`, or `rgb_shift` were used - which should be invariant to the task of object detection

### Metric Choices
- as is the custom, the chosen metric was the [`mAP`](https://cocodataset.org/#detection-eval) - which is the primary metric COCO uses. here basically the model precision is calculated at multiple IoUs(intersection over union) - `0.5:0.05:0.95` - and then averaged over the classes to finally arrive at a number
- after training for `10` epochs with only the final layers unfrozen, and then further `30` epochs with all the layers unfrozen, the `mAP` reached was `~0.29`

### Model Configuration
- all the model configurations are stored in the `src/config.py` folder as a `Config` class
- the added benefit of having them as a `python` object is that they can be very easily manipulated at runtime - like in the example training colab, the training epochs are modified at runtime

### Python Development Setup
- to help with development standard python tools like `black`, `pre-commit`, `pylint`, `mypy`, `flake8` were used

### Saved Pretrained Model
- the saved pretrained model gets accessed in the `ev_predict.ipynb`, as well as the `ConfigPredict` class in the `src/config_predict.py`
- here's the shareable google drive [link](https://drive.google.com/file/d/1vwHgzExqIyt9Ln-l9CWmMTwecULnfnUd/view?usp=sharing)
