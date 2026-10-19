"""Descriptive visual proxies only; none is an agent reward or observation."""
import numpy as np


def frame_quality(image):
    x = np.asarray(image, np.float32) / 255
    gray = x.mean(-1)
    texture = .5 * (np.abs(np.diff(gray, axis=0)).mean() +
                    np.abs(np.diff(gray, axis=1)).mean())
    hist = np.bincount(np.minimum((gray.ravel()*32).astype(int),31),minlength=32)
    p = hist[hist>0] / gray.size
    # Blue-sky heuristic, not a semantic segmentation model.
    sky = (x[...,2] > 1.15*x[...,0]) & (x[...,2] > 1.05*x[...,1]) & (x[...,2] > .3)
    return dict(frame_texture=float(texture), frame_entropy=float(-(p*np.log2(p)).sum()),
                sky_fraction=float(sky.mean()), low_detail=bool(texture < .01))
