"""CZI (Zeiss AxioScan) image loader.

Large H&E slides (tens of GB, full-res ~50000 x 30000 px) are never loaded into
memory in full. Instead we use the **pyramid levels** (downsampled copies)
embedded in the CZI to assemble a manageable whole-slide RGB image.

Typical usage::

    from foci_counter.czi_loader import CziSlide

    slide = CziSlide("dataset/20260505_Leber_HE_119.czi")
    print(slide.info())            # size, physical pixel size, available levels
    rgb = slide.read_level(9)      # whole slide downsampled 9x, (H, W, 3) uint8
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import czifile


@dataclass
class SlideInfo:
    """Summary information for a single slide."""

    path: str
    full_height: int          # full-res Y pixel count
    full_width: int           # full-res X pixel count
    pixel_size_um: float      # physical size of one full-res pixel (µm/px)
    levels: list[int]         # available downsample factors (e.g. [1, 3, 9, 27, 81])

    def level_pixel_size_um(self, downsample: int) -> float:
        """Physical pixel size (µm/px) at the given level."""
        return self.pixel_size_um * downsample

    def __str__(self) -> str:
        mm_x = self.full_width * self.pixel_size_um / 1000
        mm_y = self.full_height * self.pixel_size_um / 1000
        return (
            f"{self.path}\n"
            f"  full-res : {self.full_width} x {self.full_height} px "
            f"(~{mm_x:.1f} x {mm_y:.1f} mm)\n"
            f"  pixel    : {self.pixel_size_um:.4f} µm/px\n"
            f"  levels   : {self.levels} (downsample factors)"
        )


class CziSlide:
    """Wrapper around one CZI file giving pyramid-level access."""

    def __init__(self, path: str):
        self.path = path
        self._czi = czifile.CziFile(path)
        self._info: SlideInfo | None = None
        # Cache for the subblock-directory scan (levels + full-res extent),
        # so info() and read_level() don't re-scan on every call.
        self._levels_cache: tuple[dict[int, dict], float, float] | None = None

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "CziSlide":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._czi.close()

    # -- metadata -------------------------------------------------------
    def _pixel_size_um(self) -> float:
        """Read the X pixel size (µm/px) from the metadata XML."""
        xml = self._czi.metadata()
        m = re.search(r'<Distance Id="X">\s*<Value>([^<]+)</Value>', xml)
        if not m:
            raise ValueError(f"pixel size (ScalingX) not found: {self.path}")
        return float(m.group(1)) * 1e6  # meter -> micrometer

    def _scan_levels(self) -> tuple[dict[int, dict], float, float]:
        """Scan the subblock directory for pyramid levels and full-res extent.

        The result is cached on the instance; the (potentially slow) directory
        scan runs only once per slide.

        Returns
        -------
        levels : {downsample: {"min_x", "min_y", "max_x", "max_y"}}
        full_w, full_h : total width/height in full-res pixels
        """
        if self._levels_cache is not None:
            return self._levels_cache
        levels: dict[int, dict] = {}
        for de in self._czi.subblock_directory:
            dims = de.dims
            idx = {k: i for i, k in enumerate(dims)}
            if "X" not in idx or "Y" not in idx:
                continue
            xi, yi = idx["X"], idx["Y"]
            full_w_blk, stored_w = de.shape[xi], de.stored_shape[xi]
            if stored_w <= 0:
                continue
            ds = round(full_w_blk / stored_w)
            sx, sy = de.start[xi], de.start[yi]
            ex, ey = sx + full_w_blk, sy + de.shape[yi]
            b = levels.setdefault(
                ds, {"min_x": sx, "min_y": sy, "max_x": ex, "max_y": ey}
            )
            b["min_x"] = min(b["min_x"], sx)
            b["min_y"] = min(b["min_y"], sy)
            b["max_x"] = max(b["max_x"], ex)
            b["max_y"] = max(b["max_y"], ey)

        if not levels:
            raise ValueError(f"no X/Y subblocks found: {self.path}")

        # Full-res extent is taken from the smallest downsample factor (ds=1).
        base = levels[min(levels)]
        full_w = base["max_x"] - base["min_x"]
        full_h = base["max_y"] - base["min_y"]
        self._levels_cache = (levels, full_w, full_h)
        return self._levels_cache

    def info(self) -> SlideInfo:
        """Return the slide summary (computed once, then cached)."""
        if self._info is None:
            levels, full_w, full_h = self._scan_levels()
            self._info = SlideInfo(
                path=self.path,
                full_height=full_h,
                full_width=full_w,
                pixel_size_um=self._pixel_size_um(),
                levels=sorted(levels),
            )
        return self._info

    # -- pixel access ---------------------------------------------------
    def read_level(self, downsample: int) -> np.ndarray:
        """Assemble and return the whole-slide RGB image at a downsample factor.

        Parameters
        ----------
        downsample : int
            Must be one of ``info().levels`` (e.g. 9).

        Returns
        -------
        np.ndarray
            (H, W, 3) uint8 RGB. Empty (non-tissue) areas are filled white (255).
        """
        info = self.info()
        if downsample not in info.levels:
            raise ValueError(
                f"level {downsample} not available; choose from {info.levels}"
            )

        levels, _, _ = self._scan_levels()
        b = levels[downsample]
        min_x, min_y = b["min_x"], b["min_y"]
        out_w = int(np.ceil((b["max_x"] - min_x) / downsample))
        out_h = int(np.ceil((b["max_y"] - min_y) / downsample))

        # Start from a white background (non-tissue stays white).
        canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        for sb in self._czi.subblocks():
            de = sb.directory_entry
            dims = de.dims
            idx = {k: i for i, k in enumerate(dims)}
            if "X" not in idx or "Y" not in idx:
                continue
            xi, yi = idx["X"], idx["Y"]
            if de.stored_shape[xi] <= 0:
                continue
            if round(de.shape[xi] / de.stored_shape[xi]) != downsample:
                continue

            tile = self._to_rgb(np.asarray(sb.data()), dims)
            x0 = (de.start[xi] - min_x) // downsample
            y0 = (de.start[yi] - min_y) // downsample
            th, tw = tile.shape[:2]
            # Clip so the tile does not run past the canvas edge.
            th = min(th, out_h - y0)
            tw = min(tw, out_w - x0)
            if th <= 0 or tw <= 0:
                continue
            canvas[y0:y0 + th, x0:x0 + tw] = tile[:th, :tw]

        return canvas

    def read_region(
        self, x: int, y: int, w: int, h: int, downsample: int = 1
    ) -> np.ndarray:
        """Read a sub-region of the slide at a given pyramid level.

        Unlike ``read_level`` (which assembles the whole slide), this reads only
        the requested box -- used to fetch individual high-res tiles for nuclei
        detection without loading the full slide.

        Parameters
        ----------
        x, y, w, h : int
            Region in **full-resolution** pixel coordinates: top-left (x, y) and
            size (w, h). The box may extend past the slide; missing area stays
            white.
        downsample : int
            Pyramid level to read at (default 1 = full resolution). Must be one
            of ``info().levels``.

        Returns
        -------
        np.ndarray
            (ceil(h/downsample), ceil(w/downsample), 3) uint8 RGB.
        """
        info = self.info()
        if downsample not in info.levels:
            raise ValueError(
                f"level {downsample} not available; choose from {info.levels}"
            )
        d = downsample
        out_w = int(np.ceil(w / d))
        out_h = int(np.ceil(h / d))
        canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        x_end, y_end = x + w, y + h
        for sb in self._czi.subblocks():
            de = sb.directory_entry
            dims = de.dims
            idx = {k: i for i, k in enumerate(dims)}
            if "X" not in idx or "Y" not in idx:
                continue
            xi, yi = idx["X"], idx["Y"]
            if de.stored_shape[xi] <= 0:
                continue
            if round(de.shape[xi] / de.stored_shape[xi]) != d:
                continue

            sx, sy = de.start[xi], de.start[yi]        # block origin (full-res)
            bw, bh = de.shape[xi], de.shape[yi]        # block span (full-res)

            # Overlap between this block and the requested region (full-res).
            ox0, oy0 = max(x, sx), max(y, sy)
            ox1, oy1 = min(x_end, sx + bw), min(y_end, sy + bh)
            if ox1 <= ox0 or oy1 <= oy0:
                continue

            tile = self._to_rgb(np.asarray(sb.data()), dims)  # level-px (bh/d, bw/d, 3)
            # Source offset inside the tile and destination offset in canvas,
            # both in level pixels.
            src_x, src_y = (ox0 - sx) // d, (oy0 - sy) // d
            dst_x, dst_y = (ox0 - x) // d, (oy0 - y) // d
            cw = min((ox1 - ox0) // d, tile.shape[1] - src_x, out_w - dst_x)
            ch = min((oy1 - oy0) // d, tile.shape[0] - src_y, out_h - dst_y)
            if cw <= 0 or ch <= 0:
                continue
            canvas[dst_y:dst_y + ch, dst_x:dst_x + cw] = \
                tile[src_y:src_y + ch, src_x:src_x + cw]

        return canvas

    @staticmethod
    def _to_rgb(arr: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
        """Convert a subblock array of arbitrary axis order to (H, W, 3) uint8 RGB."""
        idx = {k: i for i, k in enumerate(dims)}
        yi, xi = idx["Y"], idx["X"]
        # Color axis: brightfield RGB is usually stored as 3 channels on the 'S'
        # (sample) axis.
        ci = idx.get("S", idx.get("C"))
        order = [yi, xi] + ([ci] if ci is not None else [])
        keep = set(order)
        # Squeeze the remaining size-1 axes (C, Z, T, ...) by taking index 0.
        sl = [slice(None) if i in keep else 0 for i in range(arr.ndim)]
        arr = arr[tuple(sl)]
        # Reorder axes to (Y, X, color).
        new_order = [a for a in [yi, xi, ci] if a is not None]
        # After the squeeze the axis count shrank, so use argsort to recover the
        # relative order only.
        rank = {ax: r for r, ax in enumerate(sorted(new_order))}
        arr = np.transpose(arr, [rank[ax] for ax in new_order])
        if arr.ndim == 2:  # grayscale -> replicate to 3 channels
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
