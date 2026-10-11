#!/usr/bin/env bash
# Build a pinned native runtime and download small.en inside ignored project dirs.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$project_dir/.whisper-runtime"
command -v cmake >/dev/null || { echo 'Install CMake first (brew install cmake).'; exit 1; }
command -v git >/dev/null || { echo 'Install Xcode command line tools first.'; exit 1; }
mkdir -p "$project_dir/models"
if [ ! -d "$runtime_dir" ]; then
  git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp.git "$runtime_dir"
fi
native_arch="$(uname -m)"
cmake -S "$runtime_dir" -B "$runtime_dir/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="$native_arch" -DGGML_NATIVE=OFF -DWHISPER_BUILD_TESTS=OFF
cmake --build "$runtime_dir/build" --config Release --target whisper-cli -j 4
if [ ! -f "$project_dir/models/ggml-small.en.bin" ]; then
  curl -fL --retry 2 https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin \
    -o "$project_dir/models/ggml-small.en.bin.part"
  mv "$project_dir/models/ggml-small.en.bin.part" "$project_dir/models/ggml-small.en.bin"
fi
echo "Whisper ready. Set FLOWSTATE_WHISPER_BIN=$runtime_dir/build/bin/whisper-cli"
echo "Set FLOWSTATE_WHISPER_MODEL=$project_dir/models/ggml-small.en.bin"
