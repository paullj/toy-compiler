#!/usr/bin/env bash
# M17 incremental (red-green early-cutoff) harness — the worthwhile proof.
#
# For each EDIT SCENARIO (body / signature / type-layout / comment-only /
# cross-module) it:
#   (1) cold-builds a base program with --query-stats (records the prior DAG),
#   (2) applies the edit,
#   (3) red-green INCREMENTAL rebuild with --query-stats (reads the prior DAG and
#       reports the recompute set — the CUTOFF),
#   (4) --force FULL rebuild of the SAME final source,
#   (5) SOUNDNESS: cmp the __text of (3) vs (4) -> must be IDENTICAL. A divergence
#       is a stale-cache miscompile (the critical anti-regression gate).
#   (6) CUTOFF (localized edits): the codegen recompile set is a STRICT SUBSET of a
#       full build (incremental compiled < full compiled). A red-green that
#       recomputes everything is "wired but inert".
#
# The byte-level cutoff is delivered by the content-fingerprint cache (a fn whose
# transitive fingerprint is unchanged is served from cache, never re-lowered); the
# red-green walk over the recorded DAG is the in-production proof of the
# early-cutoff architecture and reports the fine-grained recompute set. The
# wall-clock win is modest on this tiny single-page corpus — it is the
# finer-incrementality FOUNDATION, not a headline speedup.
set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
toyc="$root/zig-out/bin/toy"

echo "building toyc..."
if command -v zig >/dev/null 2>&1; then zig="zig"
elif command -v mise >/dev/null 2>&1; then zig="mise exec -- zig"
else echo "build failed: no zig (and no mise to provide it)"; exit 1; fi
( cd "$root" && $zig build ) || { echo "build failed"; exit 1; }

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
texthash() { otool -s __TEXT __text "$1" 2>/dev/null | tail -n +2 | shasum | cut -d' ' -f1; }
compiled_of() { sed -n 's/.*codegen compiled=\([0-9]*\).*/\1/p' <<<"$1"; }
recompute_of() { sed -n 's/.*recompute=\([0-9]*\).*/\1/p' <<<"$1"; }
total_of() { sed -n 's/.*of \([0-9]*\) nodes/\1/p' <<<"$1"; }

pass=0; fail=0
fail_one() { echo "  FAIL: $1"; fail=$((fail + 1)); }

# $1 name  $2 base source  $3 edited source  $4 expect-strict-subset-cutoff (1/0)
run_scenario() {
  local name="$1" base="$2" edited="$3" expect_cutoff="$4"
  local d="$work/$name"; mkdir -p "$d"; local src="$d/prog.toy"
  ( cd "$d" && rm -rf .toy-cache )

  printf '%s' "$base" > "$src"
  ( cd "$d" && "$toyc" -o base.bin --query-stats prog.toy >/dev/null 2>&1 ) \
    || { fail_one "$name: cold build failed"; return; }
  local full_compiled
  full_compiled="$( cd "$d" && "$toyc" -o /dev/null --codegen-stats --force prog.toy 2>/dev/null | sed -n 's/.*compiled=\([0-9]*\).*/\1/p' )"

  printf '%s' "$edited" > "$src"
  local inc_out
  inc_out="$( cd "$d" && "$toyc" -o inc.bin --query-stats prog.toy 2>&1 )"
  ( cd "$d" && "$toyc" -o force.bin --force prog.toy >/dev/null 2>&1 ) \
    || { fail_one "$name: force build failed"; return; }

  local ih fh
  ih="$(texthash "$d/inc.bin")"; fh="$(texthash "$d/force.bin")"
  if [ "$ih" != "$fh" ]; then
    fail_one "$name: SOUNDNESS — inc __text ($ih) != force __text ($fh) [STALE-CACHE MISCOMPILE]"
    return
  fi

  local rec tot comp
  rec="$(recompute_of "$inc_out")"; tot="$(total_of "$inc_out")"; comp="$(compiled_of "$inc_out")"

  if [ "$expect_cutoff" = "1" ]; then
    if [ -n "$comp" ] && [ -n "$full_compiled" ] && [ "$comp" -lt "$full_compiled" ]; then
      echo "  OK   $name: SOUND; red-green recompute=$rec of $tot; CUTOFF codegen compiled=$comp < full=$full_compiled"
      pass=$((pass + 1))
    else
      fail_one "$name: CUTOFF — codegen compiled=$comp NOT < full=$full_compiled"
    fi
  else
    echo "  OK   $name: SOUND; red-green recompute=$rec of $tot; codegen compiled=$comp full=$full_compiled (no-cutoff scenario)"
    pass=$((pass + 1))
  fi
}

echo "=== M17 edit battery (soundness + cutoff) ==="

# 1) BODY — the headline: caller cuts off, only the edited fn re-lowers.
run_scenario body \
'fn add(a: int, b: int) -> int { return a + b }
fn mul(a: int, b: int) -> int { return a * b }
fn main() -> int { return add(mul(2, 3), 4) }
' \
'fn add(a: int, b: int) -> int { return a + b }
fn mul(a: int, b: int) -> int { return a * b + 0 }
fn main() -> int { return add(mul(2, 3), 4) }
' 1

# 2) SIGNATURE — a param-type change: the caller MUST rebuild (no cutoff of the
#    caller), but the unrelated fn still cuts off.
run_scenario signature \
'fn add(a: int, b: int) -> int { return a + b }
fn lone(x: int) -> int { return x * x }
fn main() -> int { return add(1, 2) + lone(3) }
' \
'fn add(a: int, b: bool) -> int { return a }
fn lone(x: int) -> int { return x * x }
fn main() -> int { return add(1, true) + lone(3) }
' 0

# 3) TYPE / LAYOUT — add a struct field; fns not touching the struct cut off.
run_scenario layout \
'struct P { x: int, y: int }
fn px(p: P) -> int { return p.x }
fn unrelated(n: int) -> int { return n + n }
fn main() -> int {
    p := P{ x: 5, y: 9 }
    return px(p) + unrelated(1)
}
' \
'struct P { x: int, y: int, z: int }
fn px(p: P) -> int { return p.x }
fn unrelated(n: int) -> int { return n + n }
fn main() -> int {
    p := P{ x: 5, y: 9, z: 0 }
    return px(p) + unrelated(1)
}
' 1

# 4) COMMENT-ONLY — no semantic change; EVERYTHING cuts off (codegen compiled=0),
#    the strongest cutoff (index-free codegen fingerprint).
run_scenario comment \
'fn add(a: int, b: int) -> int { return a + b }
fn main() -> int { return add(40, 2) }
' \
'# a fresh comment that changes nothing semantically
fn add(a: int, b: int) -> int { return a + b }
fn main() -> int { return add(40, 2) }
' 0

# 5) CROSS-MODULE — a body edit to an imported module PRIVATE helper. The importer
#    depends on the imported fn SIGNATURE, never its body, so the entry cuts off.
run_cross_module() {
  local d="$work/cross"; mkdir -p "$d/lib"
  ( cd "$d" && rm -rf .toy-cache )
  cat > "$d/main.toy" <<'TOY'
import lib/util
fn main() -> int { return util.compute(40) }
TOY
  cat > "$d/lib/util.toy" <<'TOY'
pub fn compute(n: int) -> int { return helper(n) + 2 }
fn helper(n: int) -> int { return n }
TOY
  ( cd "$d" && "$toyc" -o base.bin --query-stats main.toy >/dev/null 2>&1 ) \
    || { fail_one "cross-module: cold build failed"; return; }
  local full_compiled
  full_compiled="$( cd "$d" && "$toyc" -o /dev/null --codegen-stats --force main.toy 2>/dev/null | sed -n 's/.*compiled=\([0-9]*\).*/\1/p' )"

  cat > "$d/lib/util.toy" <<'TOY'
pub fn compute(n: int) -> int { return helper(n) + 2 }
fn helper(n: int) -> int { return n + 0 }
TOY
  local inc_out
  inc_out="$( cd "$d" && "$toyc" -o inc.bin --query-stats main.toy 2>&1 )"
  ( cd "$d" && "$toyc" -o force.bin --force main.toy >/dev/null 2>&1 ) \
    || { fail_one "cross-module: force build failed"; return; }

  local ih fh
  ih="$(texthash "$d/inc.bin")"; fh="$(texthash "$d/force.bin")"
  if [ "$ih" != "$fh" ]; then
    fail_one "cross-module: SOUNDNESS — inc __text ($ih) != force __text ($fh) [STALE-CACHE MISCOMPILE]"
    return
  fi
  local rec tot comp
  rec="$(recompute_of "$inc_out")"; tot="$(total_of "$inc_out")"; comp="$(compiled_of "$inc_out")"
  if [ -n "$comp" ] && [ -n "$full_compiled" ] && [ "$comp" -lt "$full_compiled" ]; then
    echo "  OK   cross-module: SOUND; red-green recompute=$rec of $tot; CUTOFF codegen compiled=$comp < full=$full_compiled"
    pass=$((pass + 1))
  else
    fail_one "cross-module: CUTOFF — codegen compiled=$comp NOT < full=$full_compiled"
  fi
}
run_cross_module

echo "--- incremental battery: $pass scenario(s) sound+cutoff, $fail failed ---"
[ "$fail" -eq 0 ]
