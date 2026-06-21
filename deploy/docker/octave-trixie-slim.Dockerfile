FROM m.daocloud.io/docker.io/library/python:3.11-slim

LABEL org.opencontainers.image.title="FluxMind Octave runtime"
LABEL org.opencontainers.image.description="Local Octave runtime image for FluxMind Docker code execution on China-hosted servers"

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i \
      -e "s|http://deb.debian.org/debian-security|https://mirrors.ustc.edu.cn/debian-security|g" \
      -e "s|http://deb.debian.org/debian|https://mirrors.ustc.edu.cn/debian|g" \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends octave \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["octave", "--version"]
