FROM archlinux:latest

ENV TERM=xterm-256color
ENV HOME=/home/dev
ENV PATH=$HOME/.local/bin:$PATH

RUN pacman-key --init \
    && pacman-key --populate archlinux \
    && pacman -Syu --noconfirm archlinux-keyring

RUN pacman -Syu --noconfirm \
    base-devel git cmake ninja vim neovim fish fastfetch \
    mesa opencl-mesa llvm clang libdrm pkgconf xorg-xhost \
    && pacman -Scc --noconfirm

RUN pacman -S --noconfirm opencl-mesa clinfo

RUN useradd -m -u 1000 dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER dev
WORKDIR /raytracer

SHELL ["/usr/bin/fish", "-c"]

CMD ["fish"]
