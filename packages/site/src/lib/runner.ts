import { readFile } from './vfs';

export type RunResult = { stdout: string; exit: number };

function decode(s: string): string {
	return s
		.replace(/\\n/g, '\n')
		.replace(/\\t/g, '\t')
		.replace(/\\r/g, '\r')
		.replace(/\\"/g, '"')
		.replace(/\\\\/g, '\\');
}

// MOCK runner. toy has no browser runtime yet — this approximates a run by
// echoing the program's print("...") output so the terminal reacts to the code.
//
// When toy is compiled to wasm/WASI this is the single seam to replace: mount
// the VFS as the guest's preopen directory, instantiate the module, run the
// entry file, and return the real stdout + exit code.
export async function run(entry: string): Promise<RunResult> {
	const src = await readFile(entry);
	let stdout = '';
	const re = /print\(\s*"((?:\\.|[^"\\])*)"\s*\)/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(src))) stdout += decode(m[1]);
	return { stdout, exit: 0 };
}
