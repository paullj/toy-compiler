#!/usr/bin/env bash
# Generate a large multi-module toy project for benchmarking the compiler.
#   usage: gen.sh <out_dir> <n_modules> <fns_per_module>
# Produces N flat modules (mod_0..mod_{N-1}); each has a struct, a payload enum,
# M functions (arithmetic + a loop + a statement-if), a match helper, and a `pub
# entry`. Module k calls module k-1 (a real cross-module dependency chain); within
# a module f_i calls f_{i-1}. `main.toy` imports the last module. The static call
# graph is deep but acyclic — realistic resolve/typecheck/codegen load that scales
# linearly with N*M. (Don't RUN large outputs: the runtime call chain is N*M deep.)
set -eu
out="${1:?out_dir}"; N="${2:?n_modules}"; M="${3:?fns_per_module}"
rm -rf "$out"; mkdir -p "$out"
for k in $(seq 0 $((N - 1))); do
  f="$out/mod_$k.toy"
  {
    [ "$k" -gt 0 ] && echo "import mod_$((k - 1))"
    echo "struct S_$k { a: int, b: int }"
    echo "enum E_$k { Lo, Hi(int) }"
    for i in $(seq 0 $((M - 1))); do
      echo "fn f_${k}_${i}(n: int) -> int {"
      echo "    acc := n + $i"
      echo "    for j in 0..4 {"
      echo "        acc = acc + j * $i"
      echo "    }"
      echo "    if acc < 0 { acc = 0 - acc }"
      if [ "$i" -gt 0 ]; then
        echo "    return acc + f_${k}_$((i - 1))(n)"
      elif [ "$k" -gt 0 ]; then
        echo "    return acc + mod_$((k - 1)).entry(n)"
      else
        echo "    return acc"
      fi
      echo "}"
    done
    echo "fn tag_$k(e: E_$k) -> int { match e { .Lo -> 0, .Hi(v) -> v } }"
    echo "pub fn entry(n: int) -> int {"
    echo "    s := S_$k { a: n, b: $k }"
    echo "    return f_${k}_$((M - 1))(s.a) + tag_$k(E_$k.Hi(s.b))"
    echo "}"
  } >"$f"
done
{
  echo "import mod_$((N - 1))"
  echo "fn main() -> int { return mod_$((N - 1)).entry(7) }"
} >"$out/main.toy"
echo "generated $N modules x $M fns ($((N * M + N)) fns total) -> $out/main.toy"
