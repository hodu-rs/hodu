#!/bin/bash

docker run --rm \
    -v "$(pwd):/app" \
    -v "$HOME/.cargo/registry:/root/.cargo/registry" \
    -v "$HOME/.cargo/git:/root/.cargo/git" \
    -e RUSTC_WRAPPER="" \
    -e SCCACHE_DIR="" \
    hodu-cuda11 cargo "$@"
