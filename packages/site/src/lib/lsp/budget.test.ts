import { readFileSync } from 'node:fs';
import { brotliCompressSync, constants } from 'node:zlib';
import { expect, it } from 'vitest';
import { ToyServer } from './server';

// Budgets for what the docs site ships and how fast a check answers. They leave headroom
// over today's numbers (~190 KB, ~6 ms on an M-series laptop) so only a real regression trips.
const WASM = readFileSync(new URL('../../../static/toy-lsp.wasm', import.meta.url));
const MAX_BROTLI_BYTES = 250_000;
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

it('the shipped server stays under its brotli size budget', () => {
	const size = brotliCompressSync(WASM, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length;
	expect(size).toBeLessThan(MAX_BROTLI_BYTES);
});

it('a check of a multi-feature document stays under its latency budget', async () => {
	const server = await ToyServer.load(new WebAssembly.Module(WASM), () => {});
	server.send([{ jsonrpc: '2.0', id: 0, method: 'initialize', params: {} }]);
	const uri = 'file:///examples/budget.toy';
	server.send([{ jsonrpc: '2.0', method: 'textDocument/didOpen', params: { textDocument: { uri, version: 1, text: doc } } }]);
	const times: number[] = [];
	for (let v = 2; v < 42; v++) {
		const t0 = performance.now();
		const out = server.send([
			{ jsonrpc: '2.0', method: 'textDocument/didChange', params: { textDocument: { uri, version: v }, contentChanges: [{ text: doc + `# ${v}\n` }] } }
		]) as any[];
		times.push(performance.now() - t0);
		expect(out[0].params.diagnostics).toEqual([]);
	}
	times.sort((a, b) => a - b);
	expect(times[times.length >> 1]).toBeLessThan(MAX_MEDIAN_CHECK_MS);
});
