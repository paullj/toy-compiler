#!/usr/bin/env bash
# Correctness + determinism harness (M12, post-flip). The M12 AST→codegen path
# has been DELETED; the IR path (lex→parse→resolve→types→lower→codegen→link) is
# now the one and only backend. This harness therefore no longer diffs two paths
# — instead it asserts, for every corpus program:
#   * a runnable program: exit code + stdout match its own `# expect:` directives
#     (LOCK #1: output-identical + deterministic);
#   * an expected compile-error: the program is rejected;
#   * DETERMINISM ([C11], LOCK #7): re-lowering every fn TWICE under --verify
#     asserts per-fn FnCode byte-identity (the final signed binary carries a
#     1-byte code-sign nonce that differs every build, so we verify the codegen
#     layer rather than cmp the whole executable). Verify is self-contained and
#     does NOT depend on a primed cache.
set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
toyc="$root/zig-out/bin/toy"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "building toyc..."
# Use whatever `zig` is on PATH; fall back to `mise exec -- zig` (the documented
# invocation) so the harness builds whether run directly or via `mise exec`.
if command -v zig >/dev/null 2>&1; then
  zig="zig"
elif command -v mise >/dev/null 2>&1; then
  zig="mise exec -- zig"
else
  echo "build failed: no zig (and no mise to provide it)"; exit 1
fi
( cd "$root" && $zig build ) || { echo "build failed"; exit 1; }

pass=0; fail=0
fail_one() { echo "  FAIL: $1"; fail=$((fail + 1)); }

while IFS= read -r src; do
  rel="${src#"$root"/}"

  # Skip the IMPORT SUB-FILES of a multi-module program. A multi-module program is
  # a directory whose entry is `main.toy` (the compiler discovers the rest of the
  # graph from that one entry — see examples/modules/check.sh); its sibling/nested
  # non-entry files have no `main` and use entry-relative imports, so compiling one
  # standalone here spuriously fails. The entry `main.toy` itself still compiles
  # and is kept (it adds real determinism coverage). A file is a sub-file iff some
  # ancestor dir up to examples/ holds a `main.toy` and the file is not that entry.
  if [ "$(basename "$src")" != "main.toy" ]; then
    d="$(dirname "$src")"
    while [ "$d" != "$root/examples" ] && [ "$d" != "$root" ] && [ "$d" != "/" ]; do
      if [ -f "$d/main.toy" ]; then continue 2; fi
      d="$(dirname "$d")"
    done
  fi

  directives="$(grep -E '^# expect:' "$src" | sed -E 's/^# expect:[[:space:]]*//')"
  want_cerr=0
  grep -qE '^compile-error' <<<"$directives" && want_cerr=1

  base="$(basename "$src" .toy)"
  bin="$work/${base}.bin"
  bin2="$work/${base}.verify"

  out="$("$toyc" --force -o "$bin" "$src" 2>&1)"; rc=$?

  if [ "$want_cerr" -eq 1 ]; then
    if [ "$rc" -eq 0 ]; then
      fail_one "$rel: expected compile-error but compiled (rc=$rc)"; continue
    fi
    echo "  ok: $rel (compile-error, rejected)"; pass=$((pass + 1)); continue
  fi

  if [ "$rc" -ne 0 ]; then fail_one "$rel: compile failed: $(head -1 <<<"$out")"; continue; fi

  # Run and check against the program's own `# expect:` directives.
  stdout="$("$bin")"; exit_code=$?

  ok=1
  exp_exit="$(grep -E '^exit ' <<<"$directives" | sed -E 's/^exit[[:space:]]*//')"
  raw_stdout="$(grep -E '^stdout ' <<<"$directives" | sed -E 's/^stdout[[:space:]]*"(.*)"$/\1/')"
  exp_stdout=""; [ -n "$raw_stdout" ] && exp_stdout="$(printf '%b' "$raw_stdout")"
  if [ -n "$exp_exit" ] && [ "$exit_code" -ne "$exp_exit" ]; then
    fail_one "$rel: exit $exit_code, want $exp_exit (vs # expect:)"; ok=0
  fi
  if [ -n "$exp_stdout" ] && [ "$stdout" != "$exp_stdout" ]; then
    fail_one "$rel: stdout mismatch vs # expect:"; ok=0
    echo "    got : $(printf '%q' "$stdout")"
    echo "    want: $(printf '%q' "$exp_stdout")"
  fi

  # Determinism: --verify re-lowers every fn TWICE and asserts per-fn FnCode
  # byte-identity ([C11]). It is self-contained (does not depend on a primed
  # cache), so a Debug-build assertion failure here aborts non-zero and fails
  # the gate. (--force is intentionally omitted: it is redundant under the new
  # verify semantics, and pairing it previously masked the assertion entirely.)
  if ! "$toyc" --verify -o "$bin2" "$src" >/dev/null 2>&1; then
    fail_one "$rel: codegen NON-deterministic (--verify re-lower mismatch)"; ok=0
  fi

  [ "$ok" -eq 1 ] && { echo "  ok: $rel (correct, deterministic)"; pass=$((pass + 1)); }
# Skip examples/modules/ wholesale — those MULTI-module programs (a directory whose
# entry is main.toy, the rest reached via imports) have their own harness
# (examples/modules/check.sh). Multi-module programs OUTSIDE modules/ (e.g.
# bench/medium) are handled per-file by the sub-file guard at the top of the loop:
# their entry main.toy is compiled here, their import sub-files are skipped.
done < <(find "$root/examples" -name '*.toy' -not -path '*/modules/*' | sort)

echo "---"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
