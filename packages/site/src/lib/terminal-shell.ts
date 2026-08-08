import * as vfs from './vfs';
import { run } from './runner';

export type TermIO = { write: (s: string) => void };

const DIM = '\x1b[2m';
const RESET = '\x1b[0m';
const RED = '\x1b[31m';
const PROMPT = `${RESET}toy ${DIM}$${RESET} `;

function flatPaths(entries: vfs.Entry[], out: string[] = []): string[] {
	for (const e of entries) {
		if (e.kind === 'dir') flatPaths(e.children, out);
		else out.push(e.path);
	}
	return out;
}

// A tiny emulated shell: line editing over the raw keystroke stream plus a
// handful of commands run against the virtual filesystem. The `run` command
// goes through the (mock) runner, which becomes a real wasm/WASI execution
// later without touching this shell.
export class Shell {
	private line = '';
	private busy = false;

	constructor(private io: TermIO) {}

	start() {
		this.io.write(`toy playground ${DIM}— type \`help\` to get started${RESET}\n\n`);
		this.prompt();
	}

	private prompt() {
		this.io.write(PROMPT);
	}

	async input(data: string) {
		if (this.busy) return;
		for (const ch of data) {
			if (ch === '\r' || ch === '\n') {
				this.io.write('\n');
				const cmd = this.line.trim();
				this.line = '';
				this.busy = true;
				await this.exec(cmd);
				this.busy = false;
				this.prompt();
			} else if (ch === '\x7f' || ch === '\b') {
				if (this.line.length) {
					this.line = this.line.slice(0, -1);
					this.io.write('\b \b');
				}
			} else if (ch === '\x03') {
				this.io.write('^C\n');
				this.line = '';
				this.prompt();
			} else if (ch >= ' ') {
				this.line += ch;
				this.io.write(ch);
			}
		}
	}

	/** Invoked by the Run button — echoes a command line then runs the file. */
	async runFile(path: string) {
		if (this.busy) return;
		this.busy = true;
		this.io.write(`${DIM}$ toy run ${path}${RESET}\n`);
		await this.execRun(path);
		this.busy = false;
		this.prompt();
	}

	private async exec(cmd: string) {
		if (!cmd) return;
		const [c, ...args] = cmd.split(/\s+/);
		switch (c) {
			case 'help':
				this.io.write('commands: ls, cat <file>, run <file>, clear, help\n');
				break;
			case 'clear':
				this.io.write('\x1b[2J\x1b[H');
				break;
			case 'ls': {
				const paths = flatPaths(await vfs.readTree());
				this.io.write((paths.length ? paths.join('  ') : '(empty)') + '\n');
				break;
			}
			case 'cat': {
				if (!args[0]) {
					this.io.write('usage: cat <file>\n');
					break;
				}
				try {
					const text = await vfs.readFile(args[0]);
					this.io.write(text.endsWith('\n') ? text : text + '\n');
				} catch {
					this.io.write(`cat: ${args[0]}: no such file\n`);
				}
				break;
			}
			case 'toy':
				if (args[0] === 'run' && args[1]) await this.execRun(args[1]);
				else this.io.write('usage: toy run <file>\n');
				break;
			case 'run':
				if (args[0]) await this.execRun(args[0]);
				else this.io.write('usage: run <file>\n');
				break;
			default:
				this.io.write(`${c}: command not found\n`);
		}
	}

	private async execRun(path: string) {
		if (!(await vfs.exists(path))) {
			this.io.write(`${RED}error:${RESET} no such file: ${path}\n`);
			return;
		}
		try {
			const { stdout, exit } = await run(path);
			if (stdout) this.io.write(stdout.endsWith('\n') ? stdout : stdout + '\n');
			this.io.write(`${DIM}→ exited with code ${exit}${RESET}\n`);
		} catch {
			this.io.write(`${RED}error:${RESET} failed to run ${path}\n`);
		}
	}
}
