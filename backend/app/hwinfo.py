"""Describe the server's CPU/GPU so the page can explain which inference engine fits this machine."""
from __future__ import annotations

import os
import platform
from functools import lru_cache

import onnxruntime as ort


@lru_cache(maxsize=1)
def hardware_info() -> dict:
    model, vendor, flags = platform.processor() or "unknown", "", set()
    cores: set[tuple[str, str]] = set()
    phys = core = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key == "model name":
                    model = val
                elif key == "vendor_id":
                    vendor = val
                elif key == "flags" and not flags:
                    flags = set(val.split())
                elif key == "physical id":
                    phys = val
                elif key == "core id":
                    core = val
                    cores.add((phys, core))
    except OSError:
        pass

    if vendor == "GenuineIntel" or "intel" in model.lower():
        vendor_name = "Intel"
    elif vendor == "AuthenticAMD" or "amd" in model.lower():
        vendor_name = "AMD"
    else:
        vendor_name = vendor or platform.machine()

    providers = ort.get_available_providers()
    threads = os.cpu_count() or 0
    return {
        "model": model,
        "vendor": vendor_name,
        "threads": threads,
        "cores": len(cores) or threads,
        "avx2": "avx2" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "vnni": "avx512_vnni" in flags or "avx_vnni" in flags,
        "gpu": any(p in providers for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")),
        "ort_version": ort.__version__,
    }
