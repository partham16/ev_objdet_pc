class Config:
    """A stand-in Class for a config file
    `Config.image_size` to be used instead of hardcoding it
    """

    data_url = "https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz"
    dest_dir = "/content/ev_pc"
    force_data_download = False
    annotation_file = f"{dest_dir}/trainval/annotations/bbox-annotations.json"
    img_dir = f"{dest_dir}/trainval/images"
    class_map = ["person", "car"]
    force_parser_stats_reload = False
    parser_min_margin_ratio = 0.001
    parser_min_width_height_ratio = 0.001
    img_size = 384
    img_presize = 512
    img_bs = 24
    use_model = "faster_rcnn"
    num_workers = 4
    num_freeze_epochs = 10
    num_postfreeze_epochs = 30
    do_savemodel = True
    model_savename = (
        "fasterrcnn_r50_fpn_coco" if use_model == "faster_rcnn" else use_model + "_coco"
    )
    model_savepath = f"../models/{model_savename}_epochs-{num_freeze_epochs}-{num_postfreeze_epochs}.pth"
