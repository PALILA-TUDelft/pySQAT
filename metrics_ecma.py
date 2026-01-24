from __future__ import annotations
from typing import Any, Dict

__all__ = [
    "Loudness_ECMA418_2",
    "Loudness_ECMA418_2_from_wavfile",
    "Roughness_ECMA418_2",
    "Roughness_ECMA418_2_from_wavfile",
    "Tonality_ECMA418_2",
    "Tonality_ECMA418_2_from_wavfile",
]


def _import_wrapper(name: str):
    """Dynamic import helper with clear error message when the
    `sottek_hearing_model` dependency or wrapper modules are unavailable.
    """
    try:
        import importlib
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - keep broad error for clarity
        raise RuntimeError(
            f"Could not import module '{name}'. Ensure the project dependencies are installed. "
            "If you intend to use the Sottek hearing-model ECMA metrics, install the ``sottek_hearing_model`` "
            "package and any other runtime deps. Original error: "
            f"{exc}"
        ) from exc


def Loudness_ECMA418_2(insig, fs=None, field=0, method=1, time_skip=0, show=False, dBFS=94, export_excel=None) -> Dict[str, Any]:
    """ECMA-418-2 Loudness wrapper (delegates to SHM loudness wrapper).

    See `metrics_shm_loudness_ecma_fast.py` for implementation details.
    """
    # Import the fast wrapper dynamically so this module can be imported
    # even when the Sottek library is not installed (error deferred until use).
    mod = _import_wrapper("metrics_shm_loudness_ecma_fast")
    wrapper = getattr(mod, "shm_loudness_ecma_fast_wrapper", None)
    if wrapper is None:
        raise RuntimeError("metrics_shm_loudness_ecma_fast.shm_loudness_ecma_fast_wrapper not found")
    return wrapper(insig=insig, fs=fs, field=field, method=method, time_skip=time_skip, show=show, dBFS=dBFS, export_excel=export_excel)


def Loudness_ECMA418_2_from_wavfile(wavfilename: str, dBFS: float = 94, method: int = 1, time_skip: float = 0, show: bool = False):
    return Loudness_ECMA418_2(insig=wavfilename, fs=None, field=0, method=method, time_skip=time_skip, show=show, dBFS=dBFS)


def Roughness_ECMA418_2(insig, fs=None, field=0, method=1, time_skip=0, show=False, dBFS=94, export_excel=None) -> Dict[str, Any]:
    mod = _import_wrapper("metrics_shm_roughness_ecma")
    wrapper = getattr(mod, "shm_roughness_ecma_wrapper", None)
    if wrapper is None:
        raise RuntimeError("metrics_shm_roughness_ecma.shm_roughness_ecma_wrapper not found")
    return wrapper(insig=insig, fs=fs, field=field, method=method, time_skip=time_skip, show=show, dBFS=dBFS, export_excel=export_excel)


def Roughness_ECMA418_2_from_wavfile(wavfilename: str, dBFS: float = 94, method: int = 1, time_skip: float = 0, show: bool = False):
    return Roughness_ECMA418_2(insig=wavfilename, fs=None, field=0, method=method, time_skip=time_skip, show=show, dBFS=dBFS)


def Tonality_ECMA418_2(insig, fs=None, field=0, method=1, time_skip=0, show=False, dBFS=94, export_excel=None) -> Dict[str, Any]:
    mod = _import_wrapper("metrics_shm_tonality_ecma")
    wrapper = getattr(mod, "shm_tonality_ecma_wrapper", None)
    if wrapper is None:
        raise RuntimeError("metrics_shm_tonality_ecma.shm_tonality_ecma_wrapper not found")
    return wrapper(insig=insig, fs=fs, field=field, method=method, time_skip=time_skip, show=show, dBFS=dBFS, export_excel=export_excel)


def Tonality_ECMA418_2_from_wavfile(wavfilename: str, dBFS: float = 94, method: int = 1, time_skip: float = 0, show: bool = False):
    return Tonality_ECMA418_2(insig=wavfilename, fs=None, field=0, method=method, time_skip=time_skip, show=show, dBFS=dBFS)
