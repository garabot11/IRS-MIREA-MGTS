"""Camera keyframes and NanoDet-Plus inference, independent of ROS installation."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
from pathlib import Path
import struct

import av
import cv2
import numpy as np

from .inference import model_session, uses_cuda

LABELS = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie "
    "suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
    "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()
MODEL = Path(__file__).parent / "models" / "nanodet.onnx"
SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"


class Detector:
    def __init__(self):
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Не совпала контрольная сумма модели NanoDet")
        self.cuda = model_session(MODEL, cuda=True) if uses_cuda() else None
        if self.cuda is None:
            self.net = cv2.dnn.readNetFromONNX(str(MODEL))
        else:
            self.input_name = self.cuda.get_inputs()[0].name
            # Each grid provides class scores first, then 4 x 8 box regression values.
            outputs = sorted(self.cuda.get_outputs(), key=lambda x: (-x.shape[1], -x.shape[2]))
            self.output_names = [out.name for out in outputs]
        self.anchors = {}
        for stride in (8, 16, 32, 64):
            indices = np.indices((416 // stride, 416 // stride))[::-1].reshape(2, -1).T
            self.anchors[stride] = indices * stride + 0.5 * (stride - 1)

    def detect(self, image):
        h, w = image.shape[:2]
        ratio = 416 / max(w, h)
        nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
        ox, oy = (416 - nw) // 2, (416 - nh) // 2
        inp = np.zeros((416, 416, 3), np.float32)
        inp[oy : oy + nh, ox : ox + nw] = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (nw, nh)
        )
        inp = (inp - np.array([103.53, 116.28, 123.675], np.float32)) / np.array(
            [57.375, 57.12, 58.395], np.float32
        )
        blob = cv2.dnn.blobFromImage(inp)
        if self.cuda is None:
            self.net.setInput(blob)
            raw = self.net.forward(self.net.getUnconnectedOutLayersNames())
        else:
            raw = self.cuda.run(self.output_names, {self.input_name: blob})
        result = []
        for stride, scores, regression in zip((8, 16, 32, 64), raw[::2], raw[1::2]):
            scores = scores.reshape(-1, 80)
            classes = scores.argmax(axis=1)
            prob = scores[np.arange(len(scores)), classes]
            keep = prob >= 0.35
            if not np.any(keep):
                continue
            logits = regression.reshape(-1, 4, 8)[keep]
            weights = np.exp(logits - logits.max(axis=2, keepdims=True))
            edges = (weights @ np.arange(8)) / weights.sum(axis=2) * stride
            centers = self.anchors[stride][keep]
            rects = np.column_stack([centers - edges[:, :2], centers + edges[:, 2:]])
            rects = (rects - np.tile([ox, oy], 2)) / np.tile([nw, nh], 2)
            for rect, score, label in zip(
                np.clip(rects, 0, 1), prob[keep], classes[keep]
            ):
                if rect[2] > rect[0] and rect[3] > rect[1]:
                    result.append((rect, float(score), int(label)))
        boxes = []
        for label in sorted(set(item[2] for item in result)):
            group = [item for item in result if item[2] == label]
            xywh = [[*rect[:2], *(rect[2:] - rect[:2])] for rect, _, _ in group]
            idx = cv2.dnn.NMSBoxes(xywh, [s for _, s, _ in group], 0.35, 0.55)
            for i in np.asarray(idx, dtype=int).ravel():
                rect, score, _ = group[i]
                boxes.append(
                    dict(
                        label=LABELS[label],
                        score=round(score, 3),
                        box=rect.round(4).tolist(),
                    )
                )
        return sorted(boxes, key=lambda item: -item["score"])


class Camera:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        (self.out / "frames").mkdir(exist_ok=True)
        self.frames = []
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inspection-camera"
        )
        self.pending = deque()
        self.next_index = 0
        self.last_saved = -1e10
        self.keyframe_seen = False
        self.decoder = None
        self.detector = None
        self.decode_errors = 0
        self.clock = "header.stamp"
        self.formats = set()
        self.chosen_height = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.executor.shutdown(wait=True, cancel_futures=exc_type is not None)
        if exc_type is None:
            while self.pending:
                self.frames.append(self.pending.popleft().result())

    def save(self, img, t, acquisition=None):
        if img is None or t - self.last_saved < 0.96:
            return
        # One ordered inference worker overlaps decoding and lidar reads. At most two images are in flight.
        if len(self.pending) >= 2:
            self.frames.append(self.pending.popleft().result())
        self.pending.append(
            self.executor.submit(self._process_frame, img, t, self.next_index, acquisition)
        )
        self.next_index += 1
        self.last_saved = t

    def _process_frame(self, img, t, index, acquisition=None):
        if self.detector is None:
            self.detector = Detector()
        h, w = img.shape[:2]
        if w > 800:
            img = cv2.resize(
                img, (800, max(1, round(800 * h / w))), interpolation=cv2.INTER_AREA
            )
        file = f"frames/{index:06d}.jpg"
        if not cv2.imwrite(str(self.out / file), img, [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise OSError(f"Не удалось сохранить {file}")
        frame = dict(t=round(max(0, t), 4), file=file, detections=self.detector.detect(img))
        if acquisition:
            frame.update(acquisition, sensor_size=[w, h],
                         image_sha256=hashlib.sha256((self.out / file).read_bytes()).hexdigest())
        return frame

    def go2(self, raw, t):
        self.clock = "bag receive time (Go2 camera clock has no epoch calibration)"
        if len(raw) < 25 or raw[:2] != b"\x00\x01":
            return
        height, size = struct.unpack_from("<II", raw, 12)
        self.formats.add(height)
        # These are two encodings of the same viewpoint. Keep a single decoder per resolution.
        if size > len(raw) - 20 or height not in (360, 720):
            return
        if self.chosen_height is None:
            if height == 720:
                self.chosen_height = 720
            elif t >= 1 and 720 not in self.formats:
                self.chosen_height = 360
        if height != self.chosen_height:
            return
        data = bytes(raw[20 : 20 + size])
        if not self.keyframe_seen:
            if b"\x00\x00\x01\x67" not in data:
                return
            self.decoder = av.CodecContext.create("h264", "r")
            self.decoder.thread_count = 1
            self.keyframe_seen = True
        packet = av.Packet(data)
        packet.time_base = Fraction(1, 1000000)
        packet.pts = int(t * 1000000)
        try:
            decoded = self.decoder.decode(packet)
        except av.error.FFmpegError:
            self.decode_errors += 1
            return
        for frame in decoded:
            stamp = float(frame.time) if frame.time is not None else t
            if stamp - self.last_saved >= 0.96:
                self.save(frame.to_ndarray(format="bgr24"), stamp)

    def standard(self, m, t, received_ns):
        if t - self.last_saved < 0.96:
            return
        if hasattr(m, "format"):
            img = cv2.imdecode(np.asarray(m.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            enc = m.encoding.lower()
            channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}.get(
                enc
            )
            if channels is None:
                return
            data = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)
            img = data[:, : m.width * channels].reshape(m.height, m.width, channels)
            conv = {
                "rgb8": cv2.COLOR_RGB2BGR,
                "mono8": cv2.COLOR_GRAY2BGR,
                "rgba8": cv2.COLOR_RGBA2BGR,
                "bgra8": cv2.COLOR_BGRA2BGR,
            }
            if enc in conv:
                img = cv2.cvtColor(img, conv[enc])
        stamp = m.header.stamp
        self.save(img, t, dict(optical_frame=m.header.frame_id.strip("/"),
                               stamp_ns=stamp.sec * 10**9 + stamp.nanosec,
                               received_ns=received_ns))

    def report(self):
        return dict(
            frames=self.frames,
            model="NanoDet-Plus m 1.5x • COCO",
            model_sha256=SHA256,
            inference_frames=len(self.frames),
            detections=sum(len(f["detections"]) for f in self.frames),
            clock=self.clock,
            decoded_height=self.chosen_height,
            source_heights=sorted(self.formats),
            decode_errors=self.decode_errors,
            projection="detector boxes in image coordinates",
        )
