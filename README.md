# ev_objdet_pc

> An object detection task with only persons and cars in bounding boxes (COCO format)
--- ---

![example workflow name](https://github.com/partham16/ev_objdet_pc/workflows/Install%20on%20Python%2038/badge.svg) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/partham16/ev_objdet_pc/blob/master/LICENSE)


## Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/partham16/ev_objdet_pc/blob/master/ev_train.ipynb)

```bash
git clone https://github.com/partham16/ev_objdet_pc.git
cd ev_objdet_pc
python3.8 -m venv pyenv38
source pyenv38/bin/activate
pip install -r requirements.txt
python ev_train.py
```

or use `make full_install` for full developmental set up.

**Note:**
> in `src/config.py` the `image_bs` is set as `24` - that might cause the memory to be exhausted - **reduce** the batch size in that case.


## Check out Model Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/partham16/ev_objdet_pc/blob/master/ev_predict.ipynb)

## Check out Model *demo* - **WIP**
[A docker container deployed on Azure Container Instances](http://evobjdetpc.eastus.azurecontainer.io:8501/)

- Caveat:
    > `.png` images generally aren't supported - use `.jpg`

    > as stated, *WIP*, certain edge cases around display needs to be looked into

## Read more on design choices

[README_ML.md](https://github.com/partham16/ev_objdet_pc/blob/master/README_ML.md)
