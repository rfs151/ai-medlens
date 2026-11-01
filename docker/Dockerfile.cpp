# C++ Drogon + OpenCV DNN service. Build from the project root:
#   docker build -f docker/Dockerfile.cpp -t medlens-cpp .
# Debian trixie ships OpenCV 4.10 (>= 4.7 needed for YOLOv8/11 ONNX + NMSBoxesBatched).
FROM debian:trixie AS build
# Optional, for slow networks (e.g. mainland China):
#   --build-arg APT_MIRROR=http://mirrors.aliyun.com   --build-arg GIT_PROXY=http://127.0.0.1:7890 (with --network host)
ARG APT_MIRROR=""
ARG GIT_PROXY=""
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s#http://deb.debian.org#$APT_MIRROR#g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make git ca-certificates \
        libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev \
        libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev libopencv-dnn-dev \
    && rm -rf /var/lib/apt/lists/*

ARG DROGON_VERSION=v1.9.10
RUN git ${GIT_PROXY:+-c http.proxy=$GIT_PROXY} clone --depth 1 --branch ${DROGON_VERSION} --recurse-submodules https://github.com/drogonframework/drogon /tmp/drogon \
    && cmake -S /tmp/drogon -B /tmp/drogon/build -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
    && cmake --build /tmp/drogon/build -j"$(nproc)" && cmake --install /tmp/drogon/build \
    && rm -rf /tmp/drogon

COPY cpp_backend /src
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release && cmake --build /src/build -j"$(nproc)"

ENV MEDLENS_MODELS_DIR=/app/models MEDLENS_PORT=8081
EXPOSE 8081
WORKDIR /app
RUN cp /src/build/medlens_server /src/build/bench_detector /usr/local/bin/
CMD ["medlens_server"]
