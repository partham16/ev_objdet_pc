import streamlit as st
import numpy as np

from icevision import denormalize_imagenet, draw_pred, ClassMap
from PIL import Image
from src import Predicter
from src import ConfigPredict


@st.cache(allow_output_mutation=True)
def load_model():
    """load the model and cache"""
    predicter = Predicter()
    predicter.load_model()
    return predicter


def get_bbox(
    img,
    class_map=None,
    display_label=True,
    display_bbox=True,
):
    """get the bounding boxes"""
    # load model, set the image and get prediction
    predicter = load_model()
    predicter.img = np.array(img)
    img, pred = predicter.predict()

    img = draw_pred(
        img=img,
        pred=pred,
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        display_label=display_label,
        display_bbox=display_bbox,
    )
    img = Image.fromarray(img)
    return img


def run_app():
    """run the streamlit app"""
    st.set_option("deprecation.showfileUploaderEncoding", False)

    # label = st.sidebar.checkbox(label="Label", value=True)
    # bbox = st.sidebar.checkbox(label="Bounding Box", value=True)
    label = True
    bbox = True

    st.markdown("## A `faster-rcnn` object detection model")
    st.markdown(">  of persons and cars")
    st.markdown("### ** Drag & Drop an image**")
    uploaded_file = st.file_uploader("")  # image upload widget
    my_placeholder = st.empty()
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        my_placeholder.image(image, caption="", use_column_width=True)

    if image:
        segmented_image = get_bbox(
            image,
            class_map=ClassMap(ConfigPredict.class_map),
            display_label=label,
            display_bbox=bbox,
        )
        my_placeholder.image(segmented_image)


if __name__ == "__main__":
    run_app()
