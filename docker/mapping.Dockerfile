FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY inspection/requirements.txt .
COPY inspection/cuda-constraints.txt .
ARG INFERENCE_DEVICE=cuda
RUN case "$INFERENCE_DEVICE" in \
      cpu) cp requirements.txt /tmp/runtime.txt ;; \
      cuda) sed 's/^onnxruntime==/onnxruntime-gpu[cuda,cudnn]==/' requirements.txt > /tmp/runtime.txt ;; \
      *) exit 2 ;; \
    esac && pip install --no-cache-dir --only-binary=:all: -c cuda-constraints.txt -r /tmp/runtime.txt
WORKDIR /project
ENV PYTHONPATH=/project KONTUR_INFERENCE_DEVICE=${INFERENCE_DEVICE}
COPY inspection/ ./inspection/
COPY hackathon_solution/ ./hackathon_solution/
CMD ["python", "-m", "hackathon_solution.run", "--help"]
