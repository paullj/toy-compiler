import {
	LSPClient,
	hoverTooltips,
	serverCompletion,
	signatureHelp,
	type Transport
} from '@codemirror/lsp-client';
import { asset } from '$app/paths';
import { toyDiagnostics } from './diagnostics';

let client: LSPClient | null = null;

// One worker and one server for the whole page: every editor is a document in the same
// workspace, so a block can import another by file name.
export function toyClient(): LSPClient {
	if (client) return client;
	const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
	worker.postMessage({ wasm: new URL(asset('/toy-lsp.wasm'), location.href).href });

	const handlers = new Set<(msg: string) => void>();
	worker.onmessage = (e: MessageEvent<string>) => handlers.forEach((h) => h(e.data));
	const transport: Transport = {
		send: (msg) => worker.postMessage(msg),
		subscribe: (h) => handlers.add(h),
		unsubscribe: (h) => handlers.delete(h)
	};

	client = new LSPClient({
		rootUri: 'file:///examples/',
		extensions: [toyDiagnostics(), serverCompletion(), hoverTooltips(), signatureHelp()]
	}).connect(transport);
	return client;
}
