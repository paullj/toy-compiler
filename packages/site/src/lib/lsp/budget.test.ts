import { brotliCompressSync, constants } from 'node:zlib';
import { expect, it } from 'vitest';
import { loadServer, WASM } from './rpc.test-util';

// Budgets for what the docs site ships and how fast a check answers. They leave headroom
// over today's numbers (~464 KB with runtime safety on, up from ~326 KB before the closures
// roadmap's compiler and std; well under 1 ms per check of a std-free document, about 1 ms
// for an edit of a std/vec one and every request that follows it, on an M-series laptop)
// so only a real regression trips.
const MAX_BROTLI_BYTES = 500_000;
const MAX_MEDIAN_CHECK_MS = 25;

const doc = `pub protocol Area {
    fn area(self) -> int
}

pub struct Square { side: int }

impl Square has Area {
    fn area(self) -> int { self.side * self.side }
}

pub enum Shape { Circle(int), Square(int) }

pub fn describe(shape: Shape) -> int {
    match shape {
        .Circle(r) -> 3 * r * r,
        .Square(side) -> side * side,
    }
}

pub fn total(a: Option[int], b: Option[int]) -> Option[int] {
    x := a?
    y := b?
    Option.some(x + y)
}
`;

// Imports std, whose modules every check of the document reads as well.
const vecDoc = `import std/vec

struct Acc { n: int }
impl Acc {
    fn plus(self, k: int) -> int { self.n + k }
}

fn total(xs: Vec[int]) -> int {
    xs.fold(0, fn(a, x) { a + x })
}

fn main() -> int {
    acc := Acc { n: 1 }
    xs := [1, 2, 3, 4]
    ys := xs.map(fn(x) { x * 2 }).filter(fn(x) { x > 2 })
    count := 0
    xs.each(fn(x) { count = count + x })
    return total(ys) + acc.plus(count)
}
`;

it('the shipped server stays under its brotli size budget', () => {
	const size = brotliCompressSync(WASM, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length;
	expect(size).toBeLessThan(MAX_BROTLI_BYTES);
});

it('a check of a multi-feature document stays under its latency budget', async () => {
	const server = await loadServer();
	server.send([{ jsonrpc: '2.0', id: 0, method: 'initialize', params: {} }]);
	const uri = 'file:///examples/budget.toy';
	server.send([{ jsonrpc: '2.0', method: 'textDocument/didOpen', params: { textDocument: { uri, version: 1, text: doc } } }]);
	const times: number[] = [];
	for (let v = 2; v < 42; v++) {
		const t0 = performance.now();
		const out = server.send([
			{ jsonrpc: '2.0', method: 'textDocument/didChange', params: { textDocument: { uri, version: v }, contentChanges: [{ text: doc + `# ${v}\n` }] } }
		]);
		times.push(performance.now() - t0);
		expect(out[0].params.diagnostics).toEqual([]);
	}
	times.sort((a, b) => a - b);
	expect(times[times.length >> 1]).toBeLessThan(MAX_MEDIAN_CHECK_MS);
});

it('an edit of a std-importing document and the requests that follow stay under the latency budget', async () => {
	const server = await loadServer();
	server.send([{ jsonrpc: '2.0', id: 0, method: 'initialize', params: {} }]);
	const uri = 'file:///examples/budget-vec.toy';
	server.send([{ jsonrpc: '2.0', method: 'textDocument/didOpen', params: { textDocument: { uri, version: 1, text: vecDoc } } }]);
	const line = vecDoc.split('\n').findIndex((l) => l.includes('acc.plus(count)'));
	const at = { textDocument: { uri }, position: { line, character: vecDoc.split('\n')[line].indexOf('plus') } };
	const range = { textDocument: { uri }, range: { start: { line: 0, character: 0 }, end: { line: 20, character: 0 } } };
	const times: number[] = [];
	for (let v = 2; v < 42; v++) {
		const t0 = performance.now();
		const out = server.send([
			{ jsonrpc: '2.0', method: 'textDocument/didChange', params: { textDocument: { uri, version: v }, contentChanges: [{ text: vecDoc + `# ${v}\n` }] } },
			{ jsonrpc: '2.0', id: v, method: 'textDocument/inlayHint', params: range },
			{ jsonrpc: '2.0', id: 1000 + v, method: 'textDocument/hover', params: at }
		]);
		times.push(performance.now() - t0);
		expect(out[0].params.diagnostics).toEqual([]);
		expect(out[2].result.contents.value).toContain('fn plus(self: Acc, k: int) -> int');
	}
	times.sort((a, b) => a - b);
	expect(times[times.length >> 1]).toBeLessThan(MAX_MEDIAN_CHECK_MS);
});
