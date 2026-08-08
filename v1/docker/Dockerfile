FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  curl \
  ca-certificates \
  language-pack-ja \
  fonts-ipaexfont \
  fonts-noto-cjk \
  ghostscript \
  make \
  nodejs \
  npm \
  sudo \
  poppler-utils \
  ffmpeg \
  libatk-bridge2.0-0 \
  libatk1.0-0 \
  libdrm2 \
  libgbm1 \
  libgtk-3-0 \
  libnss3 \
  libx11-xcb1 \
  libxshmfence1 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja

RUN echo "ubuntu:ubuntu" | chpasswd && \
  usermod -aG sudo ubuntu && \
  echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER ubuntu
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
RUN uv python install 3.12.0
WORKDIR /workspace
ENV UV_LINK_MODE=copy
