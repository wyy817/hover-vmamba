import math
import numpy as np

import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph

from misc.utils import center_pad_to_shape, cropping_center, get_bounding_box
from dataloader.augs import fix_mirror_padding

# Import the original functions
from ..hovernet.targets import gen_instance_hv_map, prep_sample as hovernet_prep_sample

####
def gen_targets_with_type(ann, crop_shape, nr_types=6, **kwargs):
    """Generate the targets for the network with type prediction support."""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = cropping_center(hv_map, crop_shape)
    np_map = cropping_center(np_map, crop_shape)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


def prep_sample_with_type(data, is_batch=False, **kwargs):
    """
    Prep sample function that can handle type prediction data
    """
    return hovernet_prep_sample(data, is_batch, **kwargs)
