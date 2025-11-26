FROM --platform=linux/amd64 nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev git \
    wget lsb-release software-properties-common gnupg

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 && \
    rm llvm.sh

RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-21 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100

ENV PATH="/usr/lib/llvm-21/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add clippy rustfmt

WORKDIR /app
