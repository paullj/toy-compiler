// A small virtual filesystem for the playground.
//
// Backed by the Origin Private File System (OPFS) — the browser's real
// filesystem web API — when available, with an in-memory fallback otherwise.
// The tree it exposes is exactly the shape a WASI preopen directory wants, so
// when toy is compiled to wasm/WASI this can seed the guest's filesystem
// directly (e.g. via a WASI shim) instead of being read into the editor.

export type Entry = {
	name: string;
	kind: 'file' | 'dir';
	path: string;
	children: Entry[];
};

const canOPFS =
	typeof navigator !== 'undefined' && !!(navigator.storage && navigator.storage.getDirectory);

const segs = (path: string) => path.split('/').filter(Boolean);

function sortEntries(list: Entry[]): Entry[] {
	list.sort((a, b) => {
		if (a.kind !== b.kind) return a.kind === 'dir' ? -1 : 1;
		return a.name.localeCompare(b.name);
	});
	for (const e of list) if (e.children.length) sortEntries(e.children);
	return list;
}

// ---------------------------------------------------------------- OPFS backend

async function opfsRoot(): Promise<FileSystemDirectoryHandle> {
	return await navigator.storage.getDirectory();
}

async function opfsDir(path: string, create = false): Promise<FileSystemDirectoryHandle> {
	let dir = await opfsRoot();
	for (const s of segs(path)) dir = await dir.getDirectoryHandle(s, { create });
	return dir;
}

async function opfsParent(path: string, create = false) {
	const parts = segs(path);
	const name = parts.pop() as string;
	let dir = await opfsRoot();
	for (const s of parts) dir = await dir.getDirectoryHandle(s, { create });
	return { dir, name };
}

async function opfsWalk(dir: FileSystemDirectoryHandle, prefix: string): Promise<Entry[]> {
	const out: Entry[] = [];
	// @ts-expect-error - values() async iterator is standard but not yet in TS lib
	for await (const handle of dir.values()) {
		const path = prefix ? `${prefix}/${handle.name}` : handle.name;
		if (handle.kind === 'directory') {
			out.push({ name: handle.name, kind: 'dir', path, children: await opfsWalk(handle, path) });
		} else {
			out.push({ name: handle.name, kind: 'file', path, children: [] });
		}
	}
	return sortEntries(out);
}

// ------------------------------------------------------------- memory fallback

const memFiles = new Map<string, string>();
const memDirs = new Set<string>();

function memTree(): Entry[] {
	const root: Entry = { name: '', kind: 'dir', path: '', children: [] };
	const insert = (path: string, kind: 'file' | 'dir') => {
		const parts = segs(path);
		let node = root;
		parts.forEach((name, i) => {
			const isLast = i === parts.length - 1;
			let child = node.children.find((c) => c.name === name);
			if (!child) {
				child = {
					name,
					kind: isLast ? kind : 'dir',
					path: parts.slice(0, i + 1).join('/'),
					children: []
				};
				node.children.push(child);
			}
			node = child;
		});
	};
	for (const d of memDirs) insert(d, 'dir');
	for (const f of memFiles.keys()) insert(f, 'file');
	return sortEntries(root.children);
}

// ------------------------------------------------------------------- public API

export function backend(): 'opfs' | 'memory' {
	return canOPFS ? 'opfs' : 'memory';
}

export async function readTree(): Promise<Entry[]> {
	if (canOPFS) return opfsWalk(await opfsRoot(), '');
	return memTree();
}

export async function readFile(path: string): Promise<string> {
	if (canOPFS) {
		const { dir, name } = await opfsParent(path);
		const fh = await dir.getFileHandle(name);
		return (await fh.getFile()).text();
	}
	return memFiles.get(path) ?? '';
}

export async function writeFile(path: string, content: string): Promise<void> {
	if (canOPFS) {
		const { dir, name } = await opfsParent(path, true);
		const fh = await dir.getFileHandle(name, { create: true });
		const w = await fh.createWritable();
		await w.write(content);
		await w.close();
		return;
	}
	for (let i = 1; i < segs(path).length; i++) memDirs.add(segs(path).slice(0, i).join('/'));
	memFiles.set(path, content);
}

export async function createDir(path: string): Promise<void> {
	if (canOPFS) {
		await opfsDir(path, true);
		return;
	}
	memDirs.add(segs(path).join('/'));
}

export async function remove(path: string): Promise<void> {
	if (canOPFS) {
		const { dir, name } = await opfsParent(path);
		await dir.removeEntry(name, { recursive: true });
		return;
	}
	const p = segs(path).join('/');
	memFiles.delete(p);
	memDirs.delete(p);
	for (const f of [...memFiles.keys()]) if (f.startsWith(p + '/')) memFiles.delete(f);
	for (const d of [...memDirs]) if (d.startsWith(p + '/')) memDirs.delete(d);
}

export async function exists(path: string): Promise<boolean> {
	if (canOPFS) {
		try {
			const { dir, name } = await opfsParent(path);
			try {
				await dir.getFileHandle(name);
				return true;
			} catch {
				await dir.getDirectoryHandle(name);
				return true;
			}
		} catch {
			return false;
		}
	}
	const p = segs(path).join('/');
	return memFiles.has(p) || memDirs.has(p);
}

/** Seed a default project the first time the playground is opened. */
export async function seedIfEmpty(files: { path: string; content: string }[]): Promise<void> {
	const tree = await readTree();
	if (tree.length > 0) return;
	for (const f of files) await writeFile(f.path, f.content);
}
