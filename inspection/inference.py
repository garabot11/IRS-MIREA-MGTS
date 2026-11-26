"""ONNX inference configuration shared by the camera models."""

from functools import lru_cache
import os


def uses_cuda():
    return os.environ.get("KONTUR_INFERENCE_DEVICE", "cpu").lower() == "cuda"


@lru_cache(maxsize=1)
def _load_cuda():
    import onnxruntime as ort

    ort.preload_dlls(directory="")


def model_session(model, *, cuda=False):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if cuda:
        _load_cuda()
        providers.insert(0, ("CUDAExecutionProvider", {"use_tf32": "0"}))
    session = ort.InferenceSession(str(model), options, providers=providers)
    if cuda:
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("Не удалось активировать CUDA для " + model.name)
        session.disable_fallback()
    return session
