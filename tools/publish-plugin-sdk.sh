#!/bin/bash

# Publish hodu-plugin-sdk to crates.io

set -e

if [ -n "$(git status --porcelain)" ]; then
    echo "You have local changes!"
    exit 1
fi

echo "Publishing hodu-plugin-sdk"
cp LICENSE hodu-plugin-sdk/
pushd hodu-plugin-sdk
git add LICENSE
cargo publish --no-verify --allow-dirty
popd

echo "Cleaning local state"
git reset HEAD --hard

echo "Done!"
