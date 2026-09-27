import { readFileSync } from 'node:fs';
import { ToyServer } from './server';

export const WASM = readFileSync(new URL('../../../static/toy-lsp.wasm', import.meta.url));

// Tests speak objects; the server speaks serialized messages.
export async function loadServer() {
	const server = await ToyServer.load(new WebAssembly.Module(WASM), () => {});
	return {
		send: (messages: object[]): any[] => server.send(messages.map((m) => JSON.stringify(m))).map((r) => JSON.parse(r)),
		get memoryBytes() {
			return server.memoryBytes;
		}
	};
}
