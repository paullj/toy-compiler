#!/usr/bin/env bash
# M13 worthwhile-proof harness. Proves the IR optimization passes actually work
# AND are worthwhile (not merely "implemented + tests pass"). Three checks:
#
#   (b) DIFFERENTIAL CORRECTNESS over the FULL corpus: every .toy compiled twice
#       (-O0 and -O1, both --force), run, and asserted exit_off == exit_on AND
#       stdout_off == stdout_on. The file's `# expect: exit N`/`stdout "..."` is
#       the M12 oracle (the -O0 path == the pre-opt M12 path), and #expect is
#       checked under -O0 FIRST so a wrong expected value cannot mask a bug.
#       compile-error files must be rejected at BOTH opt levels.
#
#   (a) DUAL METRIC DROP on examples/bench/*.toy: parse --opt-stats; assert BOTH
#       ir_instrs (before->after) AND emitted aarch64 instrs (text.len/4) drop at
#       -O1 vs -O0. Smaller IR alone is insufficient -- the fold must survive to
#       machine code. >=1 bench drops for EACH metric (in fact all of them do).
#
#   (c) PER-PASS CONTRIBUTION: for each bench's TARGET pass, run -O1 with only
#       that pass (--opt=<pass>) and assert its counter > 0 (robust); then run
#       -O1 --no-opt=<pass> and assert emitted_instrs is WORSE than full -O1
#       (proves the pass is the sole load-bearing one for that win AND toggleable).
#
# ALWAYS --force: cached fns contribute 0 to --opt-stats and a stale .codegen blob
# could otherwise serve cross-opt-level garbage.
#
# Run from anywhere: resolves the repo root and prefers the prebuilt
# zig-out/bin/toy, building it via `mise exec -- zig build` if missing.
set -u

root="$(cd "$(dirname "$0")/../.." && pwd)"
toyc="$root/zig-out/bin/toy"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

zig() { ( cd "$root" && mise exec -- zig "$@" ); }

if [ ! -x "$toyc" ]; then
  echo "building toyc (mise exec -- zig build)..."
  zig build || { echo "build failed"; exit 1; }
fi
[ -x "$toyc" ] || { echo "no toyc binary at $toyc"; exit 1; }

fails=0
fail() { echo "  FAIL: $*"; fails=$((fails + 1)); }

# emitted_instrs printed by --opt-stats (works at every opt level incl -O0).
stat_field() { # <statsblock> <fieldname>  -> value (e.g. emitted_instrs, folded, ...)
  sed -nE "s/.*[[:space:]]$2=([0-9]+).*/\1/p" <<<"$1" | head -1
}

# ---------------------------------------------------------------------------
echo "=== (b) FULL-CORPUS DIFFERENTIAL  opt-ON == opt-OFF (== #expect M12 oracle) ==="
corpus_pass=0; corpus_fail=0
while IFS= read -r src; do
  rel="${src#"$root"/}"
  directives="$(grep -E '^# expect:' "$src" | sed -E 's/^# expect:[[:space:]]*//')"
  if [ -z "$directives" ]; then fail "$rel: no # expect: directive"; corpus_fail=$((corpus_fail+1)); continue; fi

  exp_exit="$(grep -E '^exit ' <<<"$directives" | sed -E 's/^exit[[:space:]]*//')"
  raw_stdout="$(grep -E '^stdout ' <<<"$directives" | sed -E 's/^stdout[[:space:]]*"(.*)"$/\1/')"
  exp_stdout=""; [ -n "$raw_stdout" ] && exp_stdout="$(printf '%b' "$raw_stdout")"
  want_cerr=0
  grep -qE '^compile-error' <<<"$directives" && want_cerr=1

  b0="$work/$(basename "$src" .toy).o0"
  b1="$work/$(basename "$src" .toy).o1"
  timeout 30 "$toyc" --force -O0 -o "$b0" "$src" >/dev/null 2>&1; c0=$?
  timeout 30 "$toyc" --force -O1 -o "$b1" "$src" >/dev/null 2>&1; c1=$?

  if [ "$want_cerr" -eq 1 ]; then
    if [ "$c0" -eq 0 ] || [ "$c1" -eq 0 ]; then
      fail "$rel: expected compile-error but compiled (O0 rc=$c0 O1 rc=$c1)"; corpus_fail=$((corpus_fail+1)); continue
    fi
    echo "  ok: $rel (compile-error rejected at O0 and O1)"; corpus_pass=$((corpus_pass+1)); continue
  fi

  if [ "$c0" -ne 0 ] || [ "$c1" -ne 0 ]; then
    fail "$rel: compile failed (O0 rc=$c0 O1 rc=$c1)"; corpus_fail=$((corpus_fail+1)); continue
  fi

  o0="$(timeout 30 "$b0")"; r0=$?
  o1="$(timeout 30 "$b1")"; r1=$?

  ok=1
  # #expect under -O0 FIRST (independent M12 oracle).
  if [ -n "$exp_exit" ] && [ "$r0" -ne "$exp_exit" ]; then fail "$rel: O0 exit $r0 != #expect $exp_exit"; ok=0; fi
  if [ -n "$exp_stdout" ] && [ "$o0" != "$exp_stdout" ]; then fail "$rel: O0 stdout != #expect"; ok=0; fi
  # opt-ON == opt-OFF differential.
  if [ "$r0" -ne "$r1" ]; then fail "$rel: exit differs O0=$r0 O1=$r1"; ok=0; fi
  if [ "$o0" != "$o1" ]; then fail "$rel: stdout differs O0 vs O1"; ok=0; fi

  if [ "$ok" -eq 1 ]; then echo "  ok: $rel (exit=$r0 O0==O1==#expect)"; corpus_pass=$((corpus_pass+1));
  else corpus_fail=$((corpus_fail+1)); fi
done < <(find "$root/examples" -name '*.toy' | sort)
echo "  corpus: $corpus_pass ok, $corpus_fail fail"

# ---------------------------------------------------------------------------
# bench -> TARGET pass, and pass -> counter name. (bash 3.2: no assoc arrays,
# so case-based lookups -- portable to /bin/bash on macOS.)
benches=(const_arith const_cmp dead_branch_true dead_branch_false redundant_local forward_chain cascade)
# For (c) we attribute each bench to the single pass that, run ALONE, both posts
# a nonzero counter AND is load-bearing for the emitted win (--no-opt regresses).
# const_cmp's cascade is unlocked by FOLDING the const icmp (branch alone can't
# fold a non-const cond), so its isolated load-bearing pass is `fold`.
target_pass() { case "$1" in
  const_arith|const_cmp|cascade) echo fold;;
  dead_branch_true|dead_branch_false) echo branch;;
  redundant_local|forward_chain) echo forward;;
  *) echo "";; esac; }
pass_counter() { case "$1" in
  fold) echo folded;; branch) echo branches;; forward) echo forwarded;; dce) echo dced;;
  *) echo "";; esac; }

echo
echo "=== (a) DUAL METRIC DROP on examples/bench (ir_instrs AND emitted both fall) ==="
printf "  %-20s %-12s %-16s %s\n" "bench" "ir B->A" "emitted O0->O1" "result"
any_ir_drop=0; any_emit_drop=0
for b in "${benches[@]}"; do
  src="$root/examples/bench/$b.toy"
  s0="$(timeout 30 "$toyc" --force --opt-stats -O0 -o "$work/$b.s0" "$src" 2>&1)"
  s1="$(timeout 30 "$toyc" --force --opt-stats -O1 -o "$work/$b.s1" "$src" 2>&1)"
  ir_before="$(sed -nE 's/.*ir_instrs=([0-9]+)->([0-9]+).*/\1/p' <<<"$s1" | head -1)"
  ir_after="$(sed -nE 's/.*ir_instrs=([0-9]+)->([0-9]+).*/\2/p' <<<"$s1" | head -1)"
  e0="$(stat_field "$s0" emitted_instrs)"
  e1="$(stat_field "$s1" emitted_instrs)"
  res="ok"
  if [ -z "$ir_before" ] || [ -z "$e0" ] || [ -z "$e1" ]; then res="FAIL(parse)"; fi
  if [ -n "$ir_after" ] && [ "$ir_after" -lt "$ir_before" ]; then any_ir_drop=1; else res="FAIL(ir)"; fi
  if [ -n "$e1" ] && [ "$e1" -lt "$e0" ]; then any_emit_drop=1; else res="FAIL(emit)"; fi
  [ "$res" != "ok" ] && fail "$b: dual-metric ($res) ir=$ir_before->$ir_after emit=$e0->$e1"
  printf "  %-20s %-12s %-16s %s\n" "$b" "$ir_before->$ir_after" "$e0->$e1" "$res"
done
[ "$any_ir_drop" -eq 1 ]   || fail "no bench dropped IR instr count"
[ "$any_emit_drop" -eq 1 ] || fail "no bench dropped emitted instr count"

echo
echo "=== (c) PER-PASS CONTRIBUTION (counter>0 for target pass; --no-opt regresses emitted) ==="
printf "  %-20s %-8s %-10s %-12s %-12s %s\n" "bench" "pass" "counter" "full-O1emit" "noopt-emit" "result"
for b in "${benches[@]}"; do
  src="$root/examples/bench/$b.toy"
  p="$(target_pass "$b")"
  cname="$(pass_counter "$p")"
  # only the target pass on
  sonly="$(timeout 30 "$toyc" --force --opt-stats "--opt=$p" -o "$work/$b.only" "$src" 2>&1)"
  cval="$(stat_field "$sonly" "$cname")"
  # full O1 emitted
  sfull="$(timeout 30 "$toyc" --force --opt-stats -O1 -o "$work/$b.full" "$src" 2>&1)"
  efull="$(stat_field "$sfull" emitted_instrs)"
  # O1 with target pass removed
  snoopt="$(timeout 30 "$toyc" --force --opt-stats -O1 "--no-opt=$p" -o "$work/$b.noopt" "$src" 2>&1)"
  enoopt="$(stat_field "$snoopt" emitted_instrs)"
  res="ok"
  if [ -z "$cval" ] || [ "$cval" -le 0 ]; then res="FAIL(counter=$cval)"; fi
  if [ -z "$efull" ] || [ -z "$enoopt" ]; then res="FAIL(parse)";
  elif [ "$enoopt" -le "$efull" ]; then res="FAIL(no-regress $enoopt<=$efull)"; fi
  [ "$res" != "ok" ] && fail "$b/$p: per-pass ($res)"
  printf "  %-20s %-8s %-10s %-12s %-12s %s\n" "$b" "$p" "$cname=$cval" "$efull" "$enoopt" "$res"
done

# DCE is exercised across the cascade benches but its standalone counter is only
# nonzero once a producer pass (fold) has created dead values. Attribute it on
# const_arith with `--opt=fold,dce`: dced>0, and `-O1 --no-opt=dce` regresses the
# emitted win (the folded const feeders survive as load/op/store words without it).
sdce="$(timeout 30 "$toyc" --force --opt-stats --opt=fold,dce -o "$work/dce.only" "$root/examples/bench/const_arith.toy" 2>&1)"
dval="$(stat_field "$sdce" dced)"
sdcefull="$(timeout 30 "$toyc" --force --opt-stats -O1 -o "$work/dce.full" "$root/examples/bench/const_arith.toy" 2>&1)"
edcefull="$(stat_field "$sdcefull" emitted_instrs)"
sdcenoopt="$(timeout 30 "$toyc" --force --opt-stats -O1 --no-opt=dce -o "$work/dce.noopt" "$root/examples/bench/const_arith.toy" 2>&1)"
edcenoopt="$(stat_field "$sdcenoopt" emitted_instrs)"
dres="ok"
if [ -z "$dval" ] || [ "$dval" -le 0 ]; then dres="FAIL(counter=$dval)"; fi
if [ -z "$edcefull" ] || [ -z "$edcenoopt" ]; then dres="FAIL(parse)";
elif [ "$edcenoopt" -le "$edcefull" ]; then dres="FAIL(no-regress $edcenoopt<=$edcefull)"; fi
[ "$dres" != "ok" ] && fail "const_arith/dce: per-pass ($dres)"
printf "  %-20s %-8s %-10s %-12s %-12s %s\n" "const_arith" "dce*" "dced=$dval" "$edcefull" "$edcenoopt" "$dres"

# ---------------------------------------------------------------------------
echo
echo "==========================================================="
if [ "$fails" -eq 0 ]; then
  echo "ALL CHECKS PASS: differential (O0==O1==#expect), dual-metric drop, per-pass contribution."
  exit 0
else
  echo "$fails CHECK(S) FAILED"
  exit 1
fi
