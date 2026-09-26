import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ToyServer } from './server';

const WASM = new URL('../../../static/toy-lsp.wasm', import.meta.url);

async function boot() {
	const server = await ToyServer.load(new WebAssembly.Module(readFileSync(WASM)));
	const [init] = server.send([{ jsonrpc: '2.0', id: 0, method: 'initialize', params: {} }]) as any[];
	expect(init.result.capabilities.hoverProvider).toBe(true);
	server.send([{ jsonrpc: '2.0', method: 'initialized', params: {} }]);
	return server;
}

const open = (uri: string, text: string, version = 1) => ({
	jsonrpc: '2.0',
	method: 'textDocument/didOpen',
	params: { textDocument: { uri, languageId: 'toy', version, text } }
});

const change = (uri: string, text: string, version: number) => ({
	jsonrpc: '2.0',
	method: 'textDocument/didChange',
	params: { textDocument: { uri, version }, contentChanges: [{ text }] }
});

function published(msgs: any[], uri: string) {
	return msgs.find((m) => m.method === 'textDocument/publishDiagnostics' && m.params.uri === uri)
		?.params.diagnostics as any[];
}

describe('toy-lsp.wasm', () => {
	it('publishes a coded type error for a broken buffer, and clears it once fixed', async () => {
		const server = await boot();
		const uri = 'file:///blocks/order.toy';
		const bad = 'fn main() -> int {\n    x: int = "no"\n    return x\n}\n';
		const diags = published(server.send([open(uri, bad)]), uri);
		expect(diags.length).toBeGreaterThan(0);
		expect(diags[0].range.start.line).toBe(1);
		expect(diags[0].message).toContain("cannot bind str to 'x' of type int");

		const good = 'fn main() -> int {\n    x: int = 1\n    return x\n}\n';
		expect(published(server.send([change(uri, good, 2)]), uri)).toEqual([]);
	});

	it('answers hover and completion in the same batch', async () => {
		const server = await boot();
		const add = 'pub fn add(a: int, b: int) -> int { return a + b }\n';
		const clean = 'file:///blocks/clean.toy';
		const typing = 'file:///blocks/typing.toy';
		const at = (uri: string, line: number, character: number) => ({ textDocument: { uri }, position: { line, character } });
		const out = server.send([
			open(clean, add + 'fn main() -> int { return add(1, 2) }\n'),
			open(typing, add + 'fn main() -> int { return add(1, ad) }\n'),
			{ jsonrpc: '2.0', id: 1, method: 'textDocument/hover', params: at(clean, 1, 27) },
			{ jsonrpc: '2.0', id: 2, method: 'textDocument/completion', params: at(typing, 1, 35) }
		]) as any[];
		expect(out.find((m) => m.id === 1).result.contents.value).toContain('fn add(int, int) -> int');
		const items = out.find((m) => m.id === 2).result as any[];
		expect(items.map((i) => i.label)).toContain('add');
	});

	it('resolves an import against another open document and bundled std', async () => {
		const server = await boot();
		server.send([open('file:///blocks/helper.toy', 'pub fn seven() -> int { return 7 }\n')]);
		const uri = 'file:///blocks/main.toy';
		const text = 'import helper\nimport std/math\nfn main() -> int { return helper.seven() }\n';
		const diags = published(server.send([open(uri, text)]), uri);
		expect(diags.filter((d) => d.severity === 1)).toEqual([]);
	});

	it('keeps its heap flat over many edits', async () => {
		const server = await boot();
		const uri = 'file:///blocks/loop.toy';
		server.send([open(uri, 'fn main() -> int { return 0 }\n')]);
		for (let v = 2; v < 50; v++) server.send([change(uri, `fn main() -> int { return ${v} }\n`, v)]);
		const warm = server.memoryBytes;
		for (let v = 50; v < 550; v++) server.send([change(uri, `fn main() -> int { return ${v} }\n`, v)]);
		expect(server.memoryBytes).toBe(warm);
	});
});
