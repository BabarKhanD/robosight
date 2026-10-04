# Data

No datasets are bundled. RoboSight runs on any image or video; the sources below are suggested for testing and evaluation. Check each license yourself before use or redistribution.

| Dataset | Use here | Link | License (verify on the source page) |
|---|---|---|---|
| COCO val2017 | Detection/segmentation sanity checks on images | https://cocodataset.org/#download | Annotations CC BY 4.0; images carry their original Flickr licenses |
| KITTI (raw / depth sample sequences) | Driving video, optional LiDAR-derived depth GT | https://www.cvlibs.net/datasets/kitti/ | CC BY-NC-SA 3.0 (non-commercial) |
| NYU Depth V2 (labeled subset) | Indoor images with metric depth GT for `evaluate_depth_spatial.py` | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html | Research dataset; no standard open license stated, follow the site's terms and cite Silberman et al. |

## Layout

```
data/
  images/          # any .jpg/.png
  videos/          # any .mp4/.avi
  nyu_mini/
    images/        # RGB: <stem>.png
    depth/         # GT depth: <stem>.npy (meters) or <stem>.png (see --gt-scale)
```

## Preparing NYU Depth V2 mini

Download `nyu_depth_v2_labeled.mat` from the link above, then export a small subset yourself (e.g. 50 images) to `images/` and `depth/` using h5py. Depth in the .mat is in meters, so save as `.npy` and use `--gt-scale 1`.

## Third-party model licenses

- YOLOv8 (Ultralytics): AGPL-3.0, so review the terms before any commercial or networked deployment.
- Depth Anything V2 Small: Apache-2.0 (the larger variants use different, non-commercial terms).
