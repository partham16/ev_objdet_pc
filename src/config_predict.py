from .config import Config as _Config


class ConfigPredict:
    """A stand-in Class for a config file used for prediction"""

    pretrained_weights_url = (
        "https://drive.google.com/uc?id=1vwHgzExqIyt9Ln-l9CWmMTwecULnfnUd"
    )
    use_model = "faster_rcnn"
    pretrained_weights_path = "./models/fasterrcnn_r50.pth"
    class_map = _Config.class_map
