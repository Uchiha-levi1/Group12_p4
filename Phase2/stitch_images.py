#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, SimilarityTransform, resize, warp


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


@dataclass
class ImageFeatures:
    path: Path
    width: int
    height: int
    preview_width: int
    preview_height: int
    downscale_matrix: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray | None


@dataclass
class Edge:
    src: int
    dst: int
    matrix: np.ndarray
    score: float
    inliers: int
    median_error: float


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch overlapping images into one or more mosaics using scikit-image."
    )
    parser.add_argument("input_dir", nargs="?", default=".")
    parser.add_argument("--output", default="stitched.png")

    parser.add_argument("--preview-max-dim", type=int, default=900)
    parser.add_argument("--n-keypoints", type=int, default=1800)

    # Relaxed defaults
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--residual-threshold", type=float, default=3.0)
    parser.add_argument("--max-rotation-deg", type=float, default=25.0)
    parser.add_argument("--max-scale-drift", type=float, default=0.2)

    parser.add_argument("--padding", type=int, default=32)
    parser.add_argument("--blend-order", type=int, default=1, choices=(0, 1, 3))

    return parser.parse_args()


def sorted_image_paths(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in SUPPORTED_EXTENSIONS:
        paths.extend(input_dir.glob(pattern))
    return sorted(set(paths), key=lambda p: p.name.lower())


def load_rgb(path: Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def build_features(path: Path, preview_max_dim: int, n_keypoints: int) -> ImageFeatures:
    image = load_rgb(path)
    h, w = image.shape[:2]

    gray = rgb2gray(image)
    scale = min(1.0, preview_max_dim / max(h, w))

    ph, pw = int(h * scale), int(w * scale)
    preview = resize(gray, (ph, pw), anti_aliasing=True, preserve_range=True)

    orb = ORB()
    try:
        orb.detect_and_extract(preview)
        k = orb.keypoints
        d = orb.descriptors
        if len(k) > n_keypoints:
            idx = np.argsort(orb.scales)[::-1][:n_keypoints]
            k, d = k[idx], d[idx]
    except RuntimeError:
        k, d = np.empty((0, 2)), None

    downscale = np.array([[pw / w, 0, 0], [0, ph / h, 0], [0, 0, 1]])

    return ImageFeatures(path, w, h, pw, ph, downscale, k, d)


def match_pair(images, i, j, args) -> Edge | None:
    a, b = images[i], images[j]
    if a.descriptors is None or b.descriptors is None:
        return None

    matches = match_descriptors(a.descriptors, b.descriptors, cross_check=True, max_ratio=0.8)
    if len(matches) < args.min_matches:
        return None

    src = a.keypoints[matches[:, 0]][:, ::-1]
    dst = b.keypoints[matches[:, 1]][:, ::-1]

    try:
        model, inliers = ransac(
            (src, dst),
            SimilarityTransform,
            min_samples=4,
            residual_threshold=args.residual_threshold,
            max_trials=600,
        )
    except ValueError:
        return None

    if inliers is None or inliers.sum() < args.min_inliers:
        return None

    if abs(np.degrees(model.rotation)) > args.max_rotation_deg:
        return None

    if not (1 - args.max_scale_drift <= model.scale <= 1 + args.max_scale_drift):
        return None

    inliers = inliers.astype(bool)
    err = np.linalg.norm(model(src[inliers]) - dst[inliers], axis=1)
    score = inliers.sum() / (1 + np.median(err))

    return Edge(i, j, model.params.copy(), score, int(inliers.sum()), float(np.median(err)))


def get_components(n, edges):
    uf = UnionFind(n)
    for e in edges:
        uf.union(e.src, e.dst)

    comps = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)

    return sorted(comps.values(), key=len, reverse=True)


def build_spanning_tree(n, edges):
    uf = UnionFind(n)
    tree = []
    for e in sorted(edges, key=lambda x: x.score, reverse=True):
        if uf.union(e.src, e.dst):
            tree.append(e)
        if len(tree) == n - 1:
            break
    if len(tree) != n - 1:
        raise RuntimeError("Component not fully connected")
    return tree


def choose_anchor(n, edges):
    scores = np.zeros(n)
    for e in edges:
        scores[e.src] += e.score
        scores[e.dst] += e.score
    return int(np.argmax(scores))


def compute_transforms(n, edges, anchor):
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e.src].append((e.dst, e.matrix))
        adj[e.dst].append((e.src, np.linalg.inv(e.matrix)))

    T = [None] * n
    T[anchor] = np.eye(3)

    stack = [anchor]
    while stack:
        cur = stack.pop()
        for nxt, mat in adj[cur]:
            if T[nxt] is None:
                T[nxt] = T[cur] @ np.linalg.inv(mat)
                stack.append(nxt)

    return T


def warp_all(images, transforms, padding, blend_order):
    corners = []
    for img, T in zip(images, transforms):
        h, w = img.height, img.width
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        pts = (T @ np.c_[pts, np.ones(4)].T).T
        pts = pts[:, :2] / pts[:, 2:3]
        corners.append(pts)

    pts = np.vstack(corners)
    min_xy = np.floor(pts.min(0))
    max_xy = np.ceil(pts.max(0))

    offset = np.array([[1, 0, -min_xy[0] + padding],
                       [0, 1, -min_xy[1] + padding],
                       [0, 0, 1]])

    H = int(max_xy[1] - min_xy[1] + 2 * padding)
    W = int(max_xy[0] - min_xy[0] + 2 * padding)

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for img, T in zip(images, transforms):
        rgb = load_rgb(img.path)
        T = offset @ T
        warped = warp(rgb, AffineTransform(matrix=T).inverse, output_shape=(H, W),
                      order=blend_order, preserve_range=True)
        mask = warp(np.ones((img.height, img.width)), AffineTransform(matrix=T).inverse,
                    output_shape=(H, W), order=0)

        canvas += warped * mask[..., None]
        weight += mask

    weight[weight == 0] = 1
    return (canvas / weight[..., None]).astype(np.uint8)


def stitch_subset(images, edges, args, out):
    tree = build_spanning_tree(len(images), edges)
    anchor = choose_anchor(len(images), tree)
    print(f"  Anchor: {images[anchor].path.name}")

    preview_T = compute_transforms(len(images), tree, anchor)
    full_T = [
        np.linalg.inv(images[anchor].downscale_matrix) @ T @ img.downscale_matrix
        for img, T in zip(images, preview_T)
    ]

    result = warp_all(images, full_T, args.padding, args.blend_order)
    io.imsave(out, result)
    print(f"  Saved → {out}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)

    paths = sorted_image_paths(input_dir)
    images = [build_features(p, args.preview_max_dim, args.n_keypoints) for p in paths]

    edges = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            e = match_pair(images, i, j, args)
            if e:
                edges.append(e)
                print(f"Matched {paths[i].name} ↔ {paths[j].name} ({e.inliers} inliers)")

    if not edges:
        print("No overlaps found")
        return

    comps = get_components(len(images), edges)
    print(f"\nFound {len(comps)} components")

    base = output.with_suffix("")

    for idx, comp in enumerate(comps):
        if len(comp) < 2:
            print(f"Skipping singleton {paths[comp[0]].name}")
            continue

        print(f"\n=== Component {idx} ({len(comp)} images) ===")

        index_map = {old: new for new, old in enumerate(comp)}
        sub_images = [images[i] for i in comp]

        sub_edges = [
            Edge(index_map[e.src], index_map[e.dst], e.matrix, e.score, e.inliers, e.median_error)
            for e in edges if e.src in index_map and e.dst in index_map
        ]

        out_path = base.parent / f"{base.name}_part{idx}.png"

        try:
            stitch_subset(sub_images, sub_edges, args, out_path)
        except RuntimeError as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
