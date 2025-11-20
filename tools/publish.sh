#!/bin/bash

# Internal crates (published first, in dependency order)
internal_crates=(
    hodu_macro_utils

    hodu_cpu_kernels
    hodu_cuda_kernels
    hodu_metal_kernels

    hodu_core
    hodu_nn/macros
    hodu_nn
    hodu_utils/macros
    hodu_utils

    hodu_internal
)

# Public crates (published after internal crates)
public_crates=(
    hodu-cli
    hodu-lib
)

if [ -n "$(git status --porcelain)" ]; then
    echo "You have local changes!"
    exit 1
fi

# Publish internal crates
pushd crates

for crate in "${internal_crates[@]}"
do
  echo "Publishing crates/${crate}"
  cp ../LICENSE "$crate"
  pushd "$crate"
  git add LICENSE
  cargo publish --no-verify --allow-dirty
  popd
  sleep 20
done

popd

# Publish public crates
for crate in "${public_crates[@]}"
do
  echo "Publishing ${crate}"
  cp LICENSE "$crate"
  pushd "$crate"
  git add LICENSE
  cargo publish --allow-dirty
  popd
  sleep 20
done

echo "Cleaning local state"
git reset HEAD --hard
