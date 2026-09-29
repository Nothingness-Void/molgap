"""Bind the exact T4-only version-3 run through the shared retry binder."""
from .bind_gpu_v2 import bind_retry


if __name__ == "__main__":
    bind_retry(3)
