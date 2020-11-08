# Motivation for replacing the default `coco` parser
#  See Issue : https://github.com/airctic/icevision/issues/467

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Hashable, List, Tuple, Union

import numpy as np
from icevision import ClassMap
from icevision.core import BBox
from icevision.parsers import Parser
from icevision.parsers.mixins import BBoxesMixin, FilepathMixin, LabelsMixin, SizeMixin
from PIL import Image, ImageStat
from tqdm import tqdm


def empty_list():
    return []


class CocoDatasetStats:
    """Calculate dataset stats"""

    # num_cats
    # num_imgs
    # num_bboxs
    # cat2name
    # class_map
    # lbl2cat
    # cat2lbl
    # img2fname
    # imgs
    # img2cat2bs
    # img2cbs
    # cat2ibs
    # avg_ncats_per_img
    # avg_nboxs_per_img
    # avg_nboxs_per_cat
    # img2sz
    # chn_means
    # chn_stds
    # avg_width
    # avg_height
    def __init__(self, f_ann: str, img_dir: Path):
        self.img_dir = img_dir
        with open(f_ann, "r") as json_f:
            ann = json.load(json_f)
        self.num_cats = len(ann["categories"])
        self.num_imgs = len(ann["images"])
        self.num_bboxs = len(ann["annotations"])

        # build cat id to name, assign FRCNN
        self.cat2name = {c["id"]: c["name"] for c in ann["categories"]}
        self.class_map = ClassMap(list(self.cat2name.values()))

        # need to translate coco subset category id to indexable label id
        # expected labels w 0 = background
        self.lbl2cat = {self.class_map.get_name(n): c for c, n in self.cat2name.items()}
        self.cat2lbl = {cat: lbl for lbl, cat in self.lbl2cat.items()}
        self.lbl2cat[0] = (0, "background")
        self.cat2lbl[0] = 0

        # img_id to file map
        self.img2fname = {img["id"]: img["file_name"] for img in ann["images"]}
        self.imgs = [
            {"id": img_id, "file_name": img_fname}
            for (img_id, img_fname) in self.img2fname.items()
        ]

        # build up some maps for later analysis
        self.img2l2bs: Dict = {}
        self.img2lbs: Dict = defaultdict(empty_list)
        self.l2ibs: Dict = defaultdict(empty_list)
        # anno_id = 0
        for a in ann["annotations"]:
            img_id = a["image_id"]
            cat_id = a["category_id"]
            lbl_id = self.cat2lbl[cat_id]
            l2bs_for_img = self.img2l2bs.get(
                img_id, {lbl: [] for lbl in range(1 + len(self.cat2name))}
            )
            (x, y, w, h) = a["bbox"]
            if w > 1 and h > 1:
                b = (x, y, w, h)
                ib = (img_id, *b)
                lb = (lbl_id, *b)
                l2bs_for_img[lbl_id].append(b)
                self.l2ibs[lbl_id].append(ib)
                self.img2lbs[img_id].append(lb)
            self.img2l2bs[img_id] = l2bs_for_img

        acc_ncats_per_img = 0.0
        acc_nboxs_per_img = 0.0
        for img_id, l2bs in self.img2l2bs.items():
            acc_ncats_per_img += len(l2bs)
            for lbl_id, bs in l2bs.items():
                acc_nboxs_per_img += len(bs)

        self.avg_ncats_per_img = acc_ncats_per_img / self.num_imgs
        self.avg_nboxs_per_img = acc_nboxs_per_img / self.num_imgs

        acc_nboxs_per_cat = 0.0
        for lbl_id, ibs in self.l2ibs.items():
            acc_nboxs_per_cat += len(ibs)

        self.avg_nboxs_per_cat = acc_nboxs_per_cat / self.num_cats

        # compute Images per channel means and std deviation using PIL.ImageStat.Stat()

        self.img2sz = {}
        n = 0
        mean = np.zeros((3,))
        stddev = np.zeros((3,))
        avgw = 0
        avgh = 0
        for img in tqdm(self.imgs):
            img_id = img["id"]
            fname = f"{img_dir}/{img['file_name']}"
            n = n + 1
            img = Image.open(fname)
            istat = ImageStat.Stat(img)
            width, height = img.size
            avgw = (width + (n - 1) * avgw) / n
            avgh = (height + (n - 1) * avgh) / n
            self.img2l2bs[img_id][0].append(
                (
                    width / 3,
                    height / 3,
                    width / 3,
                    height / 3,
                )
            )  # hack to add a backgrnd box
            mean = (istat.mean + (n - 1) * mean) / n
            stddev = (istat.stddev + (n - 1) * stddev) / n
            self.img2sz[fname] = (width, height)

        self.chn_means = mean
        self.chn_stds = stddev
        self.avg_width = avgw
        self.avg_height = avgh


def load_stats(f_ann: str, img_dir: Path, force_reload: bool = False):
    """load (or calculate) the stat"""
    stats_fpath = f"{img_dir}/stats.pkl"
    stats = None
    if os.path.isfile(stats_fpath) and not force_reload:
        try:
            stats = pickle.load(open(stats_fpath, "rb"))
        except Exception as e:
            print(f"Failed to read precomputed stats: {e}")

    if stats is None:
        stats = CocoDatasetStats(f_ann, img_dir)
        pickle.dump(stats, open(stats_fpath, "wb"))

    return stats


def box_within_bounds(
    x, y, w, h, width, height, min_margin_ratio, min_width_height_ratio
):
    """
    function for checking whether bbox width-height falls within set margin
    """
    min_width = min_width_height_ratio * width
    min_height = min_width_height_ratio * height
    if w < min_width or h < min_height:
        return False
    top_margin = min_margin_ratio * height
    bottom_margin = height - top_margin
    left_margin = min_margin_ratio * width
    right_margin = width - left_margin
    if x < left_margin or x > right_margin:
        return False
    if y < top_margin or y > bottom_margin:
        return False
    return True


class SubCocoParser(Parser, LabelsMixin, BBoxesMixin, FilepathMixin, SizeMixin):
    """
    Albumentations data augmentation requires a certain bbox width-height
    w.r.t the primary image
    This Parser ensures that we filter for that
    See Issue : https://github.com/airctic/icevision/issues/467
    """

    def __init__(
        self,
        stats: CocoDatasetStats,
        min_margin_ratio=0.15,
        min_width_height_ratio=0.1,
        quiet=True,
    ):
        self.stats = stats
        self.data = (
            []
        )  # list of tuple of form (img_id, width, height, bbox, label_id, img_path)
        skipped = 0
        for img_id, imgfname in stats.img2fname.items():
            imgf = f"{stats.img_dir}/{imgfname}"
            width, height = stats.img2sz[imgf]  # updated
            bboxs = []
            lids = []
            for lid, x, y, w, h in stats.img2lbs[img_id]:
                if lid is not None and box_within_bounds(
                    x, y, w, h, width, height, min_margin_ratio, min_width_height_ratio
                ):
                    b = [int(x), int(y), int(w), int(h)]
                    _ = int(lid)
                    bboxs.append(b)
                    lids.append(_)
                else:
                    if not quiet:
                        print(f"warning: skipping lxywh of {lid, x, y, w, h}")

            if len(bboxs) > 0:
                self.data.append(
                    (
                        img_id,
                        width,
                        height,
                        bboxs,
                        lids,
                        imgf,
                    )
                )
            else:
                skipped += 1

        print(f"Skipped {skipped} out of {stats.num_imgs} images")

    def __iter__(self):
        yield from iter(self.data)

    def __len__(self):
        return len(self.data)

    def imageid(self, o) -> Hashable:
        return o[0]

    def filepath(self, o) -> Union[str, Path]:
        return o[5]

    def height(self, o) -> int:
        return o[2]

    def width(self, o) -> int:
        return o[1]

    def image_width_height(self, o) -> Tuple[int, int]:
        return (o[1], o[2])

    def labels(self, o) -> List[int]:
        return o[4]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(x, y, w, h) for x, y, w, h in o[3]]
