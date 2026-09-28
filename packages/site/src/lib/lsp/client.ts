import {
	LSPClient,
	hoverTooltips,
	serverCompletion,
	signatureHelp,
	type Transport
} from '@codemirror/lsp-client';
import { asset } from '$app/paths';
import { toyDiagnostics } from './diagnostics';

// A server that has not answered for this long is presumed stuck (a check is ~1ms).
const HANG_MS = 5000;

// A server that has run this long without trapping has recovered.
const HEALTHY_MS = 10_000;

// Notifications that the server never answers; anything else is awaited.
const UNANSWERED = /"method":"(initialized|\$\/cancelRequest|exit)"/;

let client: LSPClient | null = null;

// One worker and one server for the whole page: every editor is a document in the same
// workspace, so a block can import another by file name.
//
// If the server traps (the worker reports it) or stops answering, the worker is replaced
// and the client reconnected: reconnecting re-sends `initialize` and re-opens every
// editor's current text, so no document state has to be replayed by hand.
export function toyClient(): LSPClient {
	if (client) return client;
	const wasm = new URL(asset('/toy-lsp.wasm'), location.href).href;
	const handlers = new Set<(msg: string) => void>();
	let worker: Worker;
	let awaitingSince = 0;
	// A document that makes the server trap is re-opened on every reconnect (after the
	// fresh server has already answered `initialize`), so restarts back off until a
	// server stays up; only then is it presumed healthy again.
	let crashes = 0;
	let spawnedAt = 0;
	let restartPending = false;

	const spawn = () => {
		worker?.terminate();
		awaitingSince = 0;
		spawnedAt = performance.now();
		worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		worker.postMessage({ wasm });
		worker.onmessage = (e: MessageEvent<string | { crashed: true }>) => {
			awaitingSince = 0;
			if (typeof e.data !== 'string') return restart();
			handlers.forEach((h) => h(e.data as string));
		};
	};

	const transport: Transport = {
		send: (msg) => {
			if (!awaitingSince && !UNANSWERED.test(msg)) awaitingSince = performance.now();
			worker.postMessage(msg);
		},
		subscribe: (h) => handlers.add(h),
		unsubscribe: (h) => handlers.delete(h)
	};

	const restart = () => {
		if (restartPending) return;
		restartPending = true;
		if (performance.now() - spawnedAt > HEALTHY_MS) crashes = 0;
		const delay = Math.min(30_000, 250 * 2 ** crashes++);
		console.warn(`[toy-lsp] restarting the language server in ${delay}ms`);
		setTimeout(() => {
			restartPending = false;
			spawn();
			client!.disconnect();
			client!.connect(transport);
		}, delay);
	};

	spawn();
	setInterval(() => {
		if (!restartPending && awaitingSince && performance.now() - awaitingSince > HANG_MS) restart();
	}, 1000);

	client = new LSPClient({
		rootUri: 'file:///examples/',
		extensions: [toyDiagnostics(), serverCompletion(), hoverTooltips(), signatureHelp()]
	}).connect(transport);
	return client;
}
