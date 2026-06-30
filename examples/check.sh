#!/usr/bin/env bash
# Validate every example against its `# expect:` annotations (see README.md).
# Builds the compiler, then for each .toy compiles it (and runs it, unless it is
# an expected compile-error) and checks the directives. Exit 0 == corpus matches.
set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
toyc="$root/zig-out/bin/toy"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "building toy..."
( cd "$root" && zig build ) || { echo "build failed"; exit 1; }

pass=0; fail=0
fail_one() { echo "  FAIL: $1"; fail=$((fail + 1)); }

while IFS= read -r src; do
  rel="${src#"$root"/}"
  directives="$(grep -E '^# expect:' "$src" | sed -E 's/^# expect:[[:space:]]*//')"
  [ -z "$directives" ] && { fail_one "$rel: no # expect: directive"; continue; }

  exp_exit="$(grep -E '^exit ' <<<"$directives" | sed -E 's/^exit[[:space:]]*//')"
  raw_stdout="$(grep -E '^stdout ' <<<"$directives" | sed -E 's/^stdout[[:space:]]*"(.*)"$/\1/')"
  exp_stdout=""; [ -n "$raw_stdout" ] && exp_stdout="$(printf '%b' "$raw_stdout")"
  want_cerr=0; exp_cerr=""
  if grep -qE '^compile-error' <<<"$directives"; then
    want_cerr=1
    exp_cerr="$(grep -E '^compile-error' <<<"$directives" | sed -E 's/^compile-error[[:space:]]*"?(.*[^"])"?$/\1/')"
  fi

  bin="$work/$(basename "$src" .toy)"
  cout="$("$toyc" -o "$bin" "$src" 2>&1)"; crc=$?

  if [ "$want_cerr" -eq 1 ]; then
    if [ "$crc" -eq 0 ]; then fail_one "$rel: expected compile-error but it compiled"; continue; fi
    if [ -n "$exp_cerr" ] && ! grep -qF "$exp_cerr" <<<"$cout"; then
      fail_one "$rel: diagnostic missing \"$exp_cerr\""; continue
    fi
    echo "  ok: $rel (compile-error)"; pass=$((pass + 1)); continue
  fi

  if [ "$crc" -ne 0 ]; then fail_one "$rel: compile failed: $(head -1 <<<"$cout")"; continue; fi

  out="$("$bin")"; rc=$?
  ok=1
  if [ -n "$exp_exit" ] && [ "$rc" -ne "$exp_exit" ]; then fail_one "$rel: exit $rc, want $exp_exit"; ok=0; fi
  if [ -n "$exp_stdout" ] && [ "$out" != "$exp_stdout" ]; then fail_one "$rel: stdout mismatch"; ok=0; fi
  [ "$ok" -eq 1 ] && { echo "  ok: $rel"; pass=$((pass + 1)); }
# `modules/` holds multi-file programs whose non-entry files have no `main` (driven
# by `examples/modules/check.sh`); `bench/` is the perf/opt corpus (driven by
# `examples/bench/opt-check.sh`). Neither belongs in this single-file expect harness.
done < <(find "$root/examples" \( -path "$root/examples/modules" -o -path "$root/examples/bench" \) -prune -o -name '*.toy' -print | sort)

echo "---"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
