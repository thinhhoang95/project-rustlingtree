from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

from vlm_ppe.schemas import CoordinateSystem

M_PER_NM = 1852.0


@dataclass(frozen=True)
class LocalProjection:
    origin_lat: float
    origin_lon: float
    crs: CRS
    to_local: Transformer
    to_wgs84: Transformer

    @classmethod
    def from_origin(cls, origin_lat: float, origin_lon: float) -> "LocalProjection":
        proj4 = (
            f"+proj=aeqd +lat_0={float(origin_lat)} +lon_0={float(origin_lon)} "
            "+datum=WGS84 +units=m +no_defs"
        )
        local_crs = CRS.from_proj4(proj4)
        wgs84 = CRS.from_epsg(4326)
        return cls(
            origin_lat=float(origin_lat),
            origin_lon=float(origin_lon),
            crs=local_crs,
            to_local=Transformer.from_crs(wgs84, local_crs, always_xy=True),
            to_wgs84=Transformer.from_crs(local_crs, wgs84, always_xy=True),
        )

    def project_nm(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_m, y_m = self.to_local.transform(np.asarray(lon_deg, dtype=float), np.asarray(lat_deg, dtype=float))
        return np.asarray(x_m, dtype=float) / M_PER_NM, np.asarray(y_m, dtype=float) / M_PER_NM

    def unproject_nm(self, x_nm: np.ndarray, y_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self.to_wgs84.transform(np.asarray(x_nm, dtype=float) * M_PER_NM, np.asarray(y_nm, dtype=float) * M_PER_NM)
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)

    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem(
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
            proj4=self.crs.to_proj4(),
        )
