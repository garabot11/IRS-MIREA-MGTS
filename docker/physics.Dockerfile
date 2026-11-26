FROM python:3.13-slim AS voxel-ray-builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY physics/voxel_rays.c /src/voxel_rays.c
RUN cc -O3 -ffp-contract=off -fPIC -shared /src/voxel_rays.c -lm -o /libkontur_voxel_rays.so

FROM python:3.13-slim
ENV MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --only-binary=:all: https://files.pythonhosted.org/packages/40/d6/68dd5a83914bbcaf32b3165993f7955979916708cb0156d1e28231e18803/numpy-2.5.5-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=824025655df929268bfe6b47ed91a00d14209a6f760847cf150be446829e1ce4 mujoco==4.0.1 Pillow==13.0.0
RUN pip install --no-cache-dir --only-binary=:all: Flask==3.1.3 waitress==3.0.2 scipy==1.18.1 opencv-python-headless==4.14.0.94
ENV LP_NUM_THREADS=2 PYTHONUNBUFFERED=1
WORKDIR /project
ENV PYTHONPATH=/project
COPY inspection/ ./inspection/
COPY hackathon_solution/ ./hackathon_solution/
COPY physics/ ./physics/
COPY --from=voxel-ray-builder /libkontur_voxel_rays.so /usr/local/lib/
CMD ["python", "-m", "hackathon_solution.v2.simulate", "--help"]
