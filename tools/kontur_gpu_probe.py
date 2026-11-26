"""Offline checks on synthetic inputs, executed inside the selected image."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def nanodet(mode):
    import onnxruntime as ort
    from inspection import inference

    model = Path(inference.__file__).parent / "models" / "nanodet.onnx"
    cuda = mode == "gpu"
    provider = "CUDAExecutionProvider" if cuda else "CPUExecutionProvider"
    # Deterministic texture in normalized NCHW coordinates, identical for both sessions.
    rng = np.random.default_rng(2026)
    frame = rng.uniform(-2, 2, (1, 3, 416, 416)).astype(np.float32)
    session = inference.model_session(model, cuda=cuda)
    feed = {session.get_inputs()[0].name: frame}
    actual = session.run(None, feed)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = "/tmp/nanodet-profile"
    options.log_severity_level = 3
    options.intra_op_num_threads = 2
    providers = [(provider, {"use_tf32": "0"})] if cuda else [provider]
    measured = ort.InferenceSession(str(model), options, providers=providers)
    measured.disable_fallback()
    profiled = measured.run(None, feed)
    profile = json.loads(Path(measured.end_profiling()).read_text())
    kernels = [e for e in profile if e.get("cat") == "Node"
               and e.get("args", {}).get("provider") == provider
               and e["args"].get("op_name") not in {"MemcpyFromHost", "MemcpyToHost"}]
    if not kernels or session.get_providers()[0] != provider:
        raise RuntimeError(f"No model kernels verified on {provider}")
    expected = [(1, n, d) for d in (80, 32) for n in (2704, 676, 169)]
    if [x.shape for x in actual] != expected:
        raise RuntimeError("Unexpected NanoDet output dimensions")
    reference = inference.model_session(model, cuda=False).run(None, feed) if cuda else profiled
    for value, baseline, traced in zip(actual, reference, profiled):
        if not np.isfinite(value).all() or np.ptp(value) <= 0:
            raise RuntimeError("Invalid NanoDet output")
        np.testing.assert_allclose(value, baseline, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(value, traced, rtol=1e-3, atol=1e-4)
    return dict(model="NanoDet", model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                provider=provider, profiled_compute_kernels=len(kernels), outputs=len(actual),
                max_reference_difference=max(float(np.max(np.abs(x-y))) for x,y in zip(actual, reference)),
                onnxruntime=ort.__version__)


def physics(mode):
    import mujoco
    from OpenGL import GL

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <light pos="0 0 3"/><camera name="test" pos="0 -2 1" xyaxes="1 0 0 0 .447 .894"/>
      <geom type="plane" size="2 2 .1" rgba=".3 .3 .3 1"/>
      <body pos="0 0 .6"><freejoint/><geom type="sphere" size=".25" rgba="1 .1 .1 1"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(80):
        mujoco.mj_step(model, data)
    if not 0.24 < data.qpos[2] < 0.60:
        raise RuntimeError("Unexpected rigid-body position")
    with mujoco.Renderer(model, height=128, width=128) as renderer:
        renderer.update_scene(data, camera="test")
        frame = renderer.render()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        name = GL.glGetString(GL.GL_RENDERER).decode()
    colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    if colors < 16 or (mode == "gpu" and "nvidia" not in vendor.lower()):
        raise RuntimeError(f"Unexpected EGL render: {vendor}, {name}, {colors} colors")
    return dict(physics="MuJoCo CPU", mujoco=mujoco.__version__, sim_time_s=round(data.time, 5),
                body_z_m=round(float(data.qpos[2]), 5), context="EGL", gl_vendor=vendor,
                gl_renderer=name, rgb_colors=colors, frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())


if __name__ == "__main__":
    name, mode = sys.argv[1:]
    print(json.dumps({"probe": name, "mode": mode, **{"nanodet": nanodet, "physics": physics}[name](mode)}))
