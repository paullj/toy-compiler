#!/usr/bin/env bash
# Validate the MULTI-MODULE corpus. Distinct from ../check.sh: each program
# is a DIRECTORY whose entry file is `main.toy`; the compiler discovers the rest
# of the module graph from that one entry, so we only ever invoke toyc on main.toy.
# The `# expect:` directives (same grammar as the single-file corpus) live in the
# entry file. Exit 0 == the module corpus matches its annotations.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
# Resolve the toyc binary without ever invoking `zig build test` (which deadlocks
# under the --listen runner); build the normal CLI binary if it is missing.
toyc="$root/zig-out/bin/toy"
if [ ! -x "$toyc" ]; then
  echo "building toyc..."
  ( cd "$root" && zig build ) || { echo "build failed"; exit 1; }
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

pass=0; fail=0
fail_one() { echo "  FAIL: $1"; fail=$((fail + 1)); }

# Every program is a directory containing an entry `main.toy`; find those entries.
while IFS= read -r entry; do
  dir="$(dirname "$entry")"
  rel="${dir#"$here"/}"
  directives="$(grep -E '^# expect:' "$entry" | sed -E 's/^# expect:[[:space:]]*//')"
  [ -z "$directives" ] && { fail_one "$rel: no # expect: directive"; continue; }

  exp_exit="$(grep -E '^exit ' <<<"$directives" | sed -E 's/^exit[[:space:]]*//')"
  raw_stdout="$(grep -E '^stdout ' <<<"$directives" | sed -E 's/^stdout[[:space:]]*"(.*)"$/\1/')"
  exp_stdout=""; [ -n "$raw_stdout" ] && exp_stdout="$(printf '%b' "$raw_stdout")"
  want_cerr=0; exp_cerr=""
  if grep -qE '^compile-error' <<<"$directives"; then
    want_cerr=1
    exp_cerr="$(grep -E '^compile-error' <<<"$directives" | sed -E 's/^compile-error[[:space:]]*"?(.*[^"])"?$/\1/')"
  fi

  bin="$work/$(basename "$dir")"
  # Run from the scratch dir with an ABSOLUTE entry path: the import root is the
  # entry file's own directory, so resolution is unaffected, while toyc's
  # `.toy-cache` lands in scratch instead of polluting the corpus.
  cout="$( cd "$work" && "$toyc" -o "$bin" "$entry" 2>&1 )"; crc=$?

  if [ "$want_cerr" -eq 1 ]; then
    if [ "$crc" -eq 0 ]; then fail_one "$rel: expected compile-error but it compiled"; continue; fi
    if [ -n "$exp_cerr" ] && ! grep -qF "$exp_cerr" <<<"$cout"; then
      fail_one "$rel: diagnostic missing \"$exp_cerr\" (got: $(head -1 <<<"$cout"))"; continue
    fi
    echo "  ok: $rel (compile-error)"; pass=$((pass + 1)); continue
  fi

  if [ "$crc" -ne 0 ]; then fail_one "$rel: compile failed: $(head -1 <<<"$cout")"; continue; fi

  out="$("$bin")"; rc=$?
  ok=1
  if [ -n "$exp_exit" ] && [ "$rc" -ne "$exp_exit" ]; then fail_one "$rel: exit $rc, want $exp_exit"; ok=0; fi
  if [ -n "$exp_stdout" ] && [ "$out" != "$exp_stdout" ]; then fail_one "$rel: stdout mismatch"; ok=0; fi
  [ "$ok" -eq 1 ] && { echo "  ok: $rel"; pass=$((pass + 1)); }
done < <(find "$here" -name 'main.toy' | sort)

echo "---"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
