from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical import canonical_json_bytes, domain_hash, sha256_bytes


_EXPECTED_JP2_SHA256 = (
    "fff7c2017a192788066a0867d78fd1ccf783216e26974b5e65287669aa472bac"
)
_EXPECTED_LBL_SHA256 = (
    "9318f41503c7d02251ed643e6dd74b737d76378be49cd492c3abb9e05d431aaa"
)
_EXPECTED_RASTERIO_VERSION = "1.5.0"
_EXPECTED_GDAL_VERSION = "3.12.1"
_EXPECTED_RASTERIO_INIT_SHA256 = (
    "08be47698f18e22a44aef866a02c72fe22572918b9bc28cb07e03c1de9bf2045"
)


def default_roi_origins() -> list[list[int]]:
    return [
        [900 + column * 500, 900 + row * 700]
        for row in range(8)
        for column in range(12)
    ]


def default_lola_contract() -> dict[str, Any]:
    return {
        "dataset_id": "LRO-L-LOLA-4-GDR-V1.0",
        "height": 7584,
        "interpolation_schema_version": "integer-bilinear-macro-only/v1",
        "jp2_sha256": _EXPECTED_JP2_SHA256,
        "lbl_sha256": _EXPECTED_LBL_SHA256,
        "map_projection_type": "POLAR STEREOGRAPHIC",
        "pixel_size_m": "20.0",
        "product_id": "LDEM_875S_20M",
        "product_version_id": "V2.1",
        "roi_origins_col_row": default_roi_origins(),
        "roi_window_size": 5,
        "sample_offset_m": "1737400.0",
        "sample_scaling_factor_m": "0.5",
        "width": 7584,
    }


def _contract(specification: dict[str, Any]) -> dict[str, Any]:
    contract = specification.get("lola_contract", default_lola_contract())
    expected = default_lola_contract()
    for field in (
        "dataset_id",
        "height",
        "interpolation_schema_version",
        "jp2_sha256",
        "lbl_sha256",
        "map_projection_type",
        "pixel_size_m",
        "product_id",
        "product_version_id",
        "roi_window_size",
        "sample_offset_m",
        "sample_scaling_factor_m",
        "width",
    ):
        if contract.get(field) != expected[field]:
            raise ValueError(f"LOLA contract drift: {field}")
    origins = contract.get("roi_origins_col_row")
    if origins != expected["roi_origins_col_row"]:
        raise ValueError("LOLA fixed ROI table drift")
    return contract


def _label_value(label_text: str, field: str) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(field)}\s*=\s*(.+?)\s*$", label_text
    )
    if match is None:
        raise ValueError(f"LOLA LBL missing {field}")
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def validate_lola_label(lbl_bytes: bytes, contract: dict[str, Any]) -> dict[str, Any]:
    try:
        text = lbl_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("LOLA LBL must be ASCII") from error
    normalized_projection = _label_value(text, "MAP_PROJECTION_TYPE").upper()
    if normalized_projection != str(contract["map_projection_type"]).upper():
        raise ValueError("LOLA projection mismatch")
    if _label_value(text, "DATA_SET_ID") != contract["dataset_id"]:
        raise ValueError("LOLA DATA_SET_ID mismatch")
    if _label_value(text, "PRODUCT_ID") != contract["product_id"]:
        raise ValueError("LOLA PRODUCT_ID mismatch")
    if _label_value(text, "PRODUCT_VERSION_ID") != contract["product_version_id"]:
        raise ValueError("LOLA PRODUCT_VERSION_ID mismatch")
    lines = int(_label_value(text, "LINES"))
    samples = int(_label_value(text, "LINE_SAMPLES"))
    if (samples, lines) != (int(contract["width"]), int(contract["height"])):
        raise ValueError("LOLA raster dimensions mismatch")
    pixel_size = Decimal(
        _label_value(text, "MAP_SCALE").split("<", 1)[0].strip()
    )
    if pixel_size != Decimal(str(contract["pixel_size_m"])):
        raise ValueError("LOLA pixel size mismatch")
    scaling = Decimal(_label_value(text, "SCALING_FACTOR"))
    offset = Decimal(_label_value(text, "OFFSET"))
    if scaling != Decimal(str(contract["sample_scaling_factor_m"])):
        raise ValueError("LOLA scaling factor mismatch")
    if offset != Decimal(str(contract["sample_offset_m"])):
        raise ValueError("LOLA offset mismatch")
    return {
        "dataset_id": contract["dataset_id"],
        "height": lines,
        "map_projection_type": normalized_projection,
        "pixel_size_m": format(pixel_size, "f"),
        "product_id": contract["product_id"],
        "product_version_id": contract["product_version_id"],
        "sample_offset_m": format(offset, "f"),
        "sample_scaling_factor_m": format(scaling, "f"),
        "width": samples,
    }


def macro_height_mm_from_raw_samples(
    samples: Sequence[Sequence[int]],
    *,
    scaling_factor_m: str,
) -> list[list[int]]:
    try:
        scale = Decimal(str(scaling_factor_m)) * Decimal(1000)
    except InvalidOperation as error:
        raise ValueError("invalid scaling_factor_m") from error
    if scale <= 0 or scale != scale.to_integral_value():
        raise ValueError("scaling factor must map integer samples to integer mm")
    scale_mm = int(scale)
    rows = [[int(value) * scale_mm for value in row] for row in samples]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("LOLA sample matrix must be non-empty and rectangular")
    return rows


def decode_lola_pair(
    jp2_bytes: bytes,
    lbl_bytes: bytes,
    specification: dict[str, Any],
) -> dict[str, Any]:
    contract = _contract(specification)
    if sha256_bytes(jp2_bytes) != contract["jp2_sha256"]:
        raise ValueError("LOLA JP2 SHA-256 mismatch")
    if sha256_bytes(lbl_bytes) != contract["lbl_sha256"]:
        raise ValueError("LOLA LBL SHA-256 mismatch")
    label = validate_lola_label(lbl_bytes, contract)
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.windows import Window
    except ImportError as error:
        raise RuntimeError("production LOLA decode requires frozen rasterio") from error
    if str(rasterio.__version__) != _EXPECTED_RASTERIO_VERSION:
        raise RuntimeError("Rasterio version drift")
    if str(rasterio.__gdal_version__) != _EXPECTED_GDAL_VERSION:
        raise RuntimeError("GDAL runtime version drift")
    rasterio_init = getattr(rasterio, "__file__", None)
    if (
        not rasterio_init
        or sha256_bytes(Path(rasterio_init).read_bytes())
        != _EXPECTED_RASTERIO_INIT_SHA256
    ):
        raise RuntimeError("Rasterio source hash drift")

    records: list[dict[str, Any]] = []
    with MemoryFile(jp2_bytes) as memory_file:
        with memory_file.open() as dataset:
            if (
                dataset.width,
                dataset.height,
                dataset.count,
                dataset.dtypes[0],
            ) != (
                int(contract["width"]),
                int(contract["height"]),
                1,
                "int16",
            ):
                raise ValueError("LOLA JP2 decoded raster contract mismatch")
            if dataset.driver != "JP2OpenJPEG":
                raise ValueError("LOLA JP2 driver mismatch")
            size = int(contract["roi_window_size"])
            for roi_index, origin in enumerate(contract["roi_origins_col_row"]):
                column, row = [int(value) for value in origin]
                array = dataset.read(
                    1, window=Window(column, row, size, size)
                )
                if array.shape != (size, size) or array.dtype != np.dtype("int16"):
                    raise ValueError(f"LOLA ROI {roi_index} decode mismatch")
                raw_samples = array.astype("<i2", copy=False).tolist()
                height_mm = macro_height_mm_from_raw_samples(
                    raw_samples,
                    scaling_factor_m=str(contract["sample_scaling_factor_m"]),
                )
                geometry_sha = domain_hash(
                    "g2-lola-roi-geometry/v1",
                    canonical_json_bytes(height_mm),
                )
                records.append(
                    {
                        "height_mm": height_mm,
                        "raw_sample_sha256": domain_hash(
                            "g2-lola-roi-raw-samples/v1",
                            np.asarray(raw_samples, dtype="<i2").tobytes(order="C"),
                        ),
                        "roi_geometry_sha256": geometry_sha,
                        "roi_index": roi_index,
                        "window_col_row_width_height": [
                            column,
                            row,
                            size,
                            size,
                        ],
                    }
                )
    roi_root = domain_hash(
        "g2-lola-roi-table/v1", canonical_json_bytes(records)
    )
    return {
        "decoder": {
            "gdal_runtime_version": _gdal_runtime_version(),
            "rasterio_version": _rasterio_version(),
        },
        "interpolation_schema_version": contract[
            "interpolation_schema_version"
        ],
        "jp2_sha256": contract["jp2_sha256"],
        "label_contract": label,
        "lbl_sha256": contract["lbl_sha256"],
        "macro_source_kind": "derived_lola_20m_macro_interpolation",
        "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "roi_records": records,
        "roi_root_sha256": roi_root,
        "schema_version": "g2-lola-decoded-roi-set/v1",
    }


def _rasterio_version() -> str:
    import rasterio

    return str(rasterio.__version__)


def _gdal_runtime_version() -> str:
    import rasterio

    return str(rasterio.__gdal_version__)
