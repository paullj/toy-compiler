#!/usr/bin/env bash
# Per-stage compile-time profile + a -j scaling sweep, using toyc's `--timings`.
#   usage: profile.sh <entry.toy> [jobs-list]   (default jobs: 1 2 4 8)
#
# Build a RELEASE toyc first for realistic ABSOLUTE numbers — a Debug build is
# dominated by safety/allocator overhead:
#   mise exec -- zig build -Doptimize=ReleaseFast
# (`--timings` reports the per-stage split: discover=lex+parse, resolve, typecheck,
#  lower=codegen+link tail, image+sign. Front-end stages are serial at any -j; only
#  `lower` scales — so `-j1` is the clean "where does the time go" breakdown.)
set -eu
toyc="$(cd "$(dirname "$0")/../.." && pwd)/zig-out/bin/toyc"
entry="${1:?usage: profile.sh <entry.toy> [jobs...]}"; shift || true
jobs=("$@"); [ "${#jobs[@]}" -eq 0 ] && jobs=(1 2 4 8)
dir="$(cd "$(dirname "$entry")" && pwd)"; base="$(basename "$entry")"
[ -x "$toyc" ] || { echo "no toyc at $toyc (build it first)"; exit 2; }

echo "== per-stage breakdown (-j 1, best of 3) =="
b=99999; out=""
for r in 1 2 3; do ( cd "$dir" && rm -rf .toy-cache ); o="$( cd "$dir" && "$toyc" -j 1 --timings --force -o /tmp/prof.bin "$base" 2>&1 )"; t="$(awk '/total/{print $2}' <<<"$o")"
  awk -v a="$t" -v c="$b" 'BEGIN{exit !(a<c)}' && { b="$t"; out="$o"; }
done
echo "$out" | sed 's/^/  /'

echo "== -j scaling (best total ms of 3) =="
for j in "${jobs[@]}"; do
  b=99999
  for r in 1 2 3; do ( cd "$dir" && rm -rf .toy-cache ); t="$( cd "$dir" && "$toyc" -j "$j" --timings --force -o /tmp/prof.bin "$base" 2>&1 | awk '/total/{print $2}' )"
    b="$(awk -v a="$t" -v c="$b" 'BEGIN{print (a<c)?a:c}')"
  done
  printf "  -j %-3s %s ms\n" "$j" "$b"
done
( cd "$dir" && rm -rf .toy-cache )
