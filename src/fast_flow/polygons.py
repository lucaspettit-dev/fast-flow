"""PolygonExtractor: turn blobs in an image into polygons.

Copied from ehd-flow's ``src/ehd_flow/polygons.py`` (same pipeline,
same defaults) and kept pure Python: OpenCV is imported lazily, so
``import fast_flow`` never requires it -- only extracting polygons
does.

Pipeline (per image layer):
  1. Binarize: values above the threshold become 1, else 0
     (127.5 for 0-255 data, 0.5 for 0-1 data).
  2. Find the external contour of every connected blob of
     1-valued pixels.  Holes are ignored, so a doughnut-shaped
     blob becomes a plain disk.
  3. Simplify each contour into a polygon (Douglas-Peucker).

On top of the extractor sits :func:`extract_layered_polygons`,
which runs it over every layer of a (ny, nx, k) image array and
returns the shape-dict format used by the force handlers:
one list per layer of ``{"vertices": [(x, y), ...]}`` dicts --
deliberately without any metadata (no charge); assigning
charges per layer is the handler's job (see its ``layermap``
constructor parameter).
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class PolygonExtractor:
    """Extracts simplified polygons from blobs in an image layer.

    Parameters mirror ehd-flow's extractor:
      epsilon_ratio -- Douglas-Peucker simplification as a fraction
                       of the contour perimeter.
      min_area      -- ignore blobs smaller than this many px^2.
    """

    THRESHOLD = 255 / 2  # 127.5

    def __init__(self, epsilon_ratio=0.002, min_area=10.0):
        if cv2 is None:
            raise ImportError(
                "opencv-python is required (pip install opencv-python)")
        self.epsilon_ratio = epsilon_ratio
        self.min_area = min_area

    # -- binarization ------------------------------------------------

    def binarize(self, layer):
        """Binarize one 2D layer -> {0, 1} uint8.

        The threshold is half the data's range: 127.5 for 0-255
        data, 0.5 for 0-1 data.
        """
        layer = np.asarray(layer)
        if layer.dtype == np.uint8:
            _, binary = cv2.threshold(layer, self.THRESHOLD, 1,
                                      cv2.THRESH_BINARY)
            return binary.astype(np.uint8)
        thresh = 0.5 if layer.size == 0 or layer.max() <= 1.0 \
            else self.THRESHOLD
        return (layer > thresh).astype(np.uint8)

    # -- polygon extraction ------------------------------------------

    def extract_polygons(self, binary, scale_x=1.0, scale_y=1.0):
        """Return a list of (name, points) polygons from a binary
        image, in pixel coordinates (x = column, y = row, scaled
        by the given factors).  Biggest blob first."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # deterministic order: biggest blob first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygons = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < self.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.epsilon_ratio * peri, True)
            pts = approx.reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts = pts.astype(float)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            polygons.append((f"blob_{i}", pts.tolist()))
        return polygons


def extract_layered_polygons(image, lx=2.0, ly=2.0, extractor=None):
    """Extract polygons from every layer of a (ny, nx, k) image.

    Returns a list with one entry per layer; each entry is a list
    of ``{"vertices": [(x, y), ...]}`` dicts (no metadata -- the
    shape format ElectrostaticForceHandler consumes together
    with its ``layermap`` parameter).

    Vertices are physical coordinates on [0, lx] x [0, ly] with
    y up: image columns map left-to-right onto x, image rows
    top-to-bottom onto y (flipped -- image row 0 is the top,
    y = ly).  A 2D array is treated as a single layer.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError("expected an image array of shape (ny, nx, k)")
    ny, nx, k = img.shape
    if extractor is None:
        extractor = PolygonExtractor()
    scale_x = lx / (nx - 1) if nx > 1 else 0.0
    scale_y = ly / (ny - 1) if ny > 1 else 0.0
    layers = []
    for c in range(k):
        binary = extractor.binarize(img[:, :, c])
        found = extractor.extract_polygons(binary)
        dicts = []
        for _name, pts in found:
            vertices = [(float(px * scale_x),
                         float(ly - py * scale_y)) for px, py in pts]
            dicts.append({"vertices": vertices})
        layers.append(dicts)
    return layers
