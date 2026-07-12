"""Runway-centred local projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyproj import CRS, Transformer  # pyright: ignore[reportMissingImports]

from .polyline import readonly_float64


@dataclass(frozen=True, init=False)
class LocalFrame:
    """Azimuthal-equidistant frame whose origin is a runway threshold."""

    origin_lat_deg: float
    origin_lon_deg: float
    proj4: str
    _to_local: Transformer = field(repr=False, compare=False)
    _to_wgs84: Transformer = field(repr=False, compare=False)

    def __init__(self, origin_lat_deg: float, origin_lon_deg: float) -> None:
        lat = float(origin_lat_deg)
        lon = float(origin_lon_deg)
        if not np.isfinite(lat) or not -90.0 <= lat <= 90.0:
            raise ValueError("origin_lat_deg must be finite and within [-90, 90]")
        if not np.isfinite(lon) or not -180.0 <= lon <= 180.0:
            raise ValueError("origin_lon_deg must be finite and within [-180, 180]")
        proj4 = (
            f"+proj=aeqd +lat_0={lat:.15g} +lon_0={lon:.15g} "
            "+datum=WGS84 +units=m +no_defs"
        )
        local_crs = CRS.from_proj4(proj4)
        wgs84 = CRS.from_epsg(4326)
        object.__setattr__(self, "origin_lat_deg", lat)
        object.__setattr__(self, "origin_lon_deg", lon)
        object.__setattr__(self, "proj4", proj4)
        object.__setattr__(self, "_to_local", Transformer.from_crs(wgs84, local_crs, always_xy=True))
        object.__setattr__(self, "_to_wgs84", Transformer.from_crs(local_crs, wgs84, always_xy=True))

    @classmethod
    def from_origin(cls, origin_lat_deg: float, origin_lon_deg: float) -> "LocalFrame":
        return cls(origin_lat_deg, origin_lon_deg)

    def project(self, lat_deg: object, lon_deg: object) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        if lat.shape != lon.shape:
            raise ValueError("lat_deg and lon_deg must have identical shapes")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("coordinates must be finite")
        east_m, north_m = self._to_local.transform(lon, lat)
        return (
            readonly_float64(east_m, name="east_m"),
            readonly_float64(north_m, name="north_m"),
        )

    def unproject(self, east_m: object, north_m: object) -> tuple[np.ndarray, np.ndarray]:
        east = np.asarray(east_m, dtype=np.float64)
        north = np.asarray(north_m, dtype=np.float64)
        if east.shape != north.shape:
            raise ValueError("east_m and north_m must have identical shapes")
        if not np.all(np.isfinite(east)) or not np.all(np.isfinite(north)):
            raise ValueError("local coordinates must be finite")
        lon, lat = self._to_wgs84.transform(east, north)
        return (
            readonly_float64(lat, name="lat_deg"),
            readonly_float64(lon, name="lon_deg"),
        )

    def project_points(self, lat_deg: object, lon_deg: object) -> np.ndarray:
        east, north = self.project(lat_deg, lon_deg)
        return readonly_float64(np.column_stack((east, north)), name="points_m", ndim=2)

    def unproject_points(self, points_m: object) -> tuple[np.ndarray, np.ndarray]:
        points = readonly_float64(points_m, name="points_m", ndim=2)
        if points.shape[1:] != (2,):
            raise ValueError("points_m must have shape (n, 2)")
        return self.unproject(points[:, 0], points[:, 1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "azimuthal_equidistant",
            "origin_lat_deg": self.origin_lat_deg,
            "origin_lon_deg": self.origin_lon_deg,
            "proj4": self.proj4,
            "units": "m",
        }


# Conceptually identical name used by the existing PPE package.
LocalProjection = LocalFrame
