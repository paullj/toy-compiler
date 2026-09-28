import { ToyServer } from './server';

// Messages that arrive while the server is busy (or still loading) queue up and are fed
// as one batch: the server then sees a `$/cancelRequest` alongside the request it cancels.
let server: ToyServer | null = null;
const queue: string[] = [];
let scheduled = false;

async function load(url: string) {
	const res = fetch(url);
	const module = await WebAssembly.compileStreaming(res).catch(async () =>
		WebAssembly.compile(await (await fetch(url)).arrayBuffer())
	);
	server = await ToyServer.load(module, (line) => console.warn('[toy-lsp]', line));
	flush();
}

function flush() {
	scheduled = false;
	if (!server || queue.length === 0) return;
	let replies: string[];
	try {
		replies = server.send(queue.splice(0));
	} catch (err) {
		// A trap leaves the wasm instance unusable; the page replaces this worker.
		console.error('[toy-lsp]', err);
		server = null;
		postMessage({ crashed: true });
		return;
	}
	for (const reply of replies) postMessage(reply);
}

self.onmessage = (e: MessageEvent<string | { wasm: string }>) => {
	if (typeof e.data !== 'string') {
		void load(e.data.wasm);
		return;
	}
	queue.push(e.data);
	if (!scheduled && server) {
		scheduled = true;
		setTimeout(flush, 0);
	}
};
