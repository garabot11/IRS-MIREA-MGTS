"""Cross-session appearance hypotheses. Image features supply no range measurements."""

from pathlib import Path

import cv2
import numpy as np

from .geometry import relative


def descriptors(directory, frames, times, poses):
    orb = cv2.ORB_create(nfeatures=550)
    previous = None
    selected = []
    for f in frames:
        i = np.searchsorted(times, f["t"]).clip(0, len(times) - 1)
        p = poses[i]
        if previous is not None:
            d = relative(previous, p)
            if np.linalg.norm(d[:2]) < 0.9 and abs(d[2]) < 0.45:
                continue
        image = cv2.imread(str(Path(directory) / f["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (400, 225), interpolation=cv2.INTER_AREA)
        keys, desc = orb.detectAndCompute(image, None)
        if desc is None or len(keys) < 30:
            continue
        selected.append((p, np.float32([k.pt for k in keys]), desc))
        previous = p
    return selected


def visual_pairs(reference_dir, reference_result, current_dir, data, poses):
    refposes = np.array([[p["x"], p["y"], p["yaw"]] for p in reference_result["poses"]])
    times = np.array([p["t"] for p in reference_result["poses"]])
    a = descriptors(
        reference_dir, reference_result["camera"]["frames"], times, refposes
    )
    b = descriptors(
        current_dir,
        data["camera"]["frames"],
        np.array([s.t for s in data["scans"]]),
        poses,
    )
    if not a or not b:
        return []
    # A Hamming vocabulary ranks candidate images before pairwise geometric verification.
    rng = np.random.default_rng(15)
    pool = np.vstack([entry[2] for entry in a])
    words = pool[rng.choice(len(pool), min(420, len(pool)), replace=False)]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def histogram(desc):
        ids = [m.trainIdx for m in matcher.match(desc, words)]
        h = np.bincount(ids, minlength=len(words)).astype(float)
        return h / max(1, h.sum())

    hist = np.array([histogram(x[2]) for x in a])
    pairs = []
    for p, xy, desc in b:
        other = np.argsort(np.abs(hist - histogram(desc)).sum(axis=1))[:8]
        best = None
        for j in other:
            rp, ref_xy, ref_desc = a[j]
            matches = [
                m
                for m, n in matcher.knnMatch(desc, ref_desc, k=2)
                if m.distance < 0.76 * n.distance
            ]
            if len(matches) < 18:
                continue
            src = np.float32([xy[m.queryIdx] for m in matches])
            dst = np.float32([ref_xy[m.trainIdx] for m in matches])
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=700)
            if H is None:
                continue
            good = inliers.ravel().astype(bool)
            n = int(good.sum())
            if n < 17 or n < 0.53 * len(matches):
                continue
            hull = cv2.contourArea(cv2.convexHull(src[good]))
            if hull < 1400:
                continue
            shift = np.median(dst[good] - src[good], axis=0)
            if abs(shift[0]) > 65 or abs(shift[1]) > 45:
                continue
            if best is None or n > best[0]:
                best = (n, p, rp)
        if best is not None:
            pairs.append(best)
    return pairs
