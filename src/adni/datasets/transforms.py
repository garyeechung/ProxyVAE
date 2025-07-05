from typing import List, Optional

from monai.transforms import Compose, MapTransform
from monai.transforms import ResizeWithPadOrCropd, Transposed
import nibabel as nib
import numpy as np
import scipy as sp


class LoadImageFromPathd(MapTransform):

    def __init__(self, keys: List[str]):
        super().__init__(keys=keys)

    def __call__(self, data):
        """
        Args:
            data: suppose data is a dict with keys as specified in self.keys,
                    and values are full paths to a nii.gz file.
        """
        d = dict(data)
        for key in self.keys:
            path = d[key]
            nii_data = nib.load(path)
            volume = nii_data.get_fdata(dtype=np.float32)
            volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
            volume = np.clip(volume, 0.0, 1.0)
            d[key] = volume
            d["zooms"] = nii_data.header.get_zooms()
        return d


class ToResolutiond(MapTransform):
    def __init__(self, keys, target_zooms: Optional[List[float]] = [1.0, 1.0, 1.0],
                 skip_zooming_depth: bool = True):
        """
        Resize the volume to the target zooms.
        Originally for converting to isotropic voxels.
        Now deprecated, using other tools to register the volume to a template space.
        Args:
            keys: keys in the data dict to apply this transform.
            target_zooms: target zooms for the volume
                          e.g., [1.0, 1.0, 1.0] is isotropic voxels at 1 mm^3.
            skip_zooming_depth: if True, the depth dimension won't be resized.
                                recommended for 2D slices.
        """
        super().__init__(keys=keys)
        self.target_zooms = target_zooms
        self.skip_zooming_depth = skip_zooming_depth

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            volume = d[key]
            zooms = d["zooms"]
            if self.skip_zooming_depth:
                target_zoom = self.target_zooms[:2]
                target_zoom.append(zooms[2])
            zoom_factors = [old_zoom / target_zoom
                            for target_zoom, old_zoom in zip(self.target_zooms, zooms)]
            volume = sp.ndimage.zoom(volume, zoom_factors, order=1)
            volume = np.clip(volume, 0.0, 1.0)
            d[key] = volume
        return d


class OneHotEncoded(MapTransform):
    def __init__(self, keys: List[str], num_classes: List[int]):
        super().__init__(keys=keys)
        self.num_classes = num_classes
        assert len(keys) == len(num_classes), "keys and num_classes must have the same length"

    def __call__(self, data):
        d = dict(data)
        for key, nc in zip(self.keys, self.num_classes):
            lab = d[key]
            d[key] = np.eye(nc)[int(lab)]
        return d


class GetCenterSliced(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys=keys)

    def __call__(self, data):
        """
        Args:
            data: suppose data is a dict with keys as specified in self.keys,
                    and values are 3D volumes.
        """
        d = dict(data)
        for key in self.keys:
            volume = d[key]
            nb_slices = volume.shape[-1]
            center_slice = nb_slices // 2
            d[key] = volume[..., center_slice:center_slice + 1]  # default to center slice
        return d


class GetRandomSliced(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys=keys)

    def __call__(self, data):
        """
        Args:
            data: suppose data is a dict with keys as specified in self.keys,
                    and values are 3D volumes.
        """
        d = dict(data)
        for key in self.keys:
            volume = d[key]
            nb_slices = volume.shape[-1]
            selected_slice = np.random.randint(0, nb_slices)
            d[key] = volume[..., selected_slice:selected_slice + 1]
        return d


class GetCenterSlicesd(MapTransform):
    def __init__(self, keys: List[str],
                 slice_range_from_center: float = 1.0):
        super().__init__(keys=keys)
        self.slice_range_from_center = slice_range_from_center
        assert 0.0 <= slice_range_from_center <= 1.0, \
            "slice_range_from_center must be in [0, 1]"

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            volume = d[key]
            nb_slices = volume.shape[-1]
            center_slice = nb_slices // 2
            slice_range = int(self.slice_range_from_center * nb_slices / 2)
            start_slice = max(0, center_slice - slice_range)
            end_slice = min(nb_slices, center_slice + slice_range)
            if start_slice >= end_slice:
                volume = volume[..., center_slice:center_slice + 1]
            else:
                volume = volume[..., start_slice:end_slice]
            volume = np.clip(volume, 0.0, 1.0)
            d[key] = volume
        return d


def get_cache_transforms(
        image_keys: List[str],
        spatial_size: Optional[List[int]] = [256, 256],
        slice_range_from_center: float = 0.03
) -> Compose:
    """
    Returns a Compose of transforms to load and preprocess the volume data.
    Note that target_zooms * spatial_size will be the actual spatial sizes in mm
    """

    transforms = [
        LoadImageFromPathd(keys=image_keys),
        Transposed(keys=["image"], indices=(2, 1, 0)),  # from HWD to DWH
        ResizeWithPadOrCropd(keys=image_keys, spatial_size=spatial_size),
        Transposed(keys=["image"], indices=(2, 1, 0)),  # from DWH back to HWD
        GetCenterSlicesd(keys=image_keys, slice_range_from_center=slice_range_from_center)
    ]
    return Compose(transforms)


def is_foreground(x):
    return x > 0.0
