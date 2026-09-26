// The server makes no filesystem calls once its disk access is off, so the WASI surface
// it links against is answered with ENOSYS except for the few calls a check really makes:
// the clock, randomness, and stderr (panic/log output).
const ENOSYS = 52;
const EBADF = 8;
const SUCCESS = 0;

export function wasiImports(memory: () => WebAssembly.Memory, onStderr: (line: string) => void) {
	const decoder = new TextDecoder();
	let pending = '';

	const view = () => new DataView(memory().buffer);

	const stub = new Proxy(
		{},
		{
			get: (_t, name) => () => {
				if (typeof name === 'string' && name.startsWith('fd_')) return EBADF;
				return ENOSYS;
			}
		}
	) as Record<string, (...args: number[]) => number>;

	const impl: Record<string, (...args: never[]) => number | bigint> = {
		clock_time_get(_id: number, _precision: bigint, out: number) {
			const ns = BigInt(Math.round(performance.now() * 1e6));
			view().setBigUint64(out, ns, true);
			return SUCCESS;
		},
		clock_res_get(_id: number, out: number) {
			view().setBigUint64(out, 1000n, true);
			return SUCCESS;
		},
		random_get(ptr: number, len: number) {
			crypto.getRandomValues(new Uint8Array(memory().buffer, ptr, len));
			return SUCCESS;
		},
		environ_sizes_get(count: number, size: number) {
			view().setUint32(count, 0, true);
			view().setUint32(size, 0, true);
			return SUCCESS;
		},
		environ_get() {
			return SUCCESS;
		},
		fd_write(fd: number, iovs: number, iovsLen: number, written: number) {
			if (fd !== 1 && fd !== 2) return EBADF;
			const dv = view();
			let total = 0;
			for (let i = 0; i < iovsLen; i++) {
				const ptr = dv.getUint32(iovs + i * 8, true);
				const len = dv.getUint32(iovs + i * 8 + 4, true);
				pending += decoder.decode(new Uint8Array(memory().buffer, ptr, len));
				total += len;
			}
			let nl: number;
			while ((nl = pending.indexOf('\n')) >= 0) {
				onStderr(pending.slice(0, nl));
				pending = pending.slice(nl + 1);
			}
			dv.setUint32(written, total, true);
			return SUCCESS;
		}
	};

	return {
		wasi_snapshot_preview1: new Proxy(impl, {
			get: (t, name: string) => t[name] ?? stub[name]
		})
	};
}
