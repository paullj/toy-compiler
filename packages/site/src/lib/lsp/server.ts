import { wasiImports } from './wasi';

type Exports = {
	memory: WebAssembly.Memory;
	toy_lsp_alloc(len: number): number;
	toy_lsp_feed(ptr: number, len: number): number;
	toy_lsp_output(): number;
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// The toy language server compiled to wasm32-wasi. `send` takes serialized JSON-RPC
// messages and returns every message the server wrote in reply (responses and
// notifications), still serialized: the editor's transport speaks strings, so nothing
// here parses JSON. The server speaks LSP base-protocol framing; this class hides it.
export class ToyServer {
	private constructor(private readonly x: Exports) {}

	static async load(module: WebAssembly.Module, onStderr: (line: string) => void = console.warn) {
		let memory: WebAssembly.Memory | undefined;
		const instance = await WebAssembly.instantiate(
			module,
			wasiImports(() => memory!, onStderr)
		);
		const x = instance.exports as unknown as Exports;
		memory = x.memory;
		return new ToyServer(x);
	}

	send(messages: string[]): string[] {
		const parts = messages.map((m) => {
			const body = encoder.encode(m);
			return [encoder.encode(`Content-Length: ${body.length}\r\n\r\n`), body];
		});
		const len = parts.flat().reduce((n, p) => n + p.length, 0);
		const ptr = this.x.toy_lsp_alloc(len);
		if (ptr === 0) throw new Error('toy-lsp: out of memory');
		let at = ptr;
		for (const p of parts.flat()) {
			new Uint8Array(this.x.memory.buffer, at, p.length).set(p);
			at += p.length;
		}
		const outLen = this.x.toy_lsp_feed(ptr, len);
		if (outLen < 0) throw new Error('toy-lsp: fatal server fault');
		const out = new Uint8Array(this.x.memory.buffer, this.x.toy_lsp_output(), outLen).slice();
		return unframe(out);
	}

	get memoryBytes() {
		return this.x.memory.buffer.byteLength;
	}
}

function unframe(bytes: Uint8Array): string[] {
	const msgs: string[] = [];
	let i = 0;
	while (i < bytes.length) {
		const headerEnd = indexOfCrlfCrlf(bytes, i);
		if (headerEnd < 0) break;
		const header = decoder.decode(bytes.subarray(i, headerEnd));
		const m = /Content-Length:\s*(\d+)/i.exec(header);
		if (!m) break;
		const start = headerEnd + 4;
		const end = start + Number(m[1]);
		msgs.push(decoder.decode(bytes.subarray(start, end)));
		i = end;
	}
	return msgs;
}

function indexOfCrlfCrlf(b: Uint8Array, from: number) {
	for (let i = from; i + 3 < b.length; i++) {
		if (b[i] === 13 && b[i + 1] === 10 && b[i + 2] === 13 && b[i + 3] === 10) return i;
	}
	return -1;
}
