// Minimal tokenizer for the `toy` language. Good enough for docs snippets —
// not a real lexer. Returns tokens with a CSS class for coloring.
export type Token = { text: string; cls: string };

const KEYWORDS = new Set([
	'fn', 'struct', 'enum', 'impl', 'has', 'protocol', 'type', 'match',
	'if', 'else', 'return', 'self', 'mut', 'true', 'false', 'import', 'pub'
]);

const TYPES = new Set([
	'int', 'uint', 'byte', 'char', 'str', 'bool', 'float',
	'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
	'Option', 'Result'
]);

const PATTERNS: [RegExp, (m: string) => string][] = [
	[/^#[^\n]*/, () => 'tok-com'],
	[/^"(?:\\.|[^"\\])*"/, () => 'tok-str'],
	[/^'(?:\\.|[^'\\])*'/, () => 'tok-str'],
	[/^0x[0-9a-fA-F_]+/, () => 'tok-num'],
	[/^\d[\d_]*(?:\.\d[\d_]*)?/, () => 'tok-num'],
	[/^[A-Za-z_][A-Za-z0-9_]*/, (m) =>
		KEYWORDS.has(m) ? 'tok-kw' : TYPES.has(m) ? 'tok-type' : ''],
	[/^\s+/, () => ''],
	[/^[{}()[\],;:.]/, () => 'tok-punc'],
	[/^[-+*/%<>=!&|?]+/, () => 'tok-punc']
];

export function tokenize(src: string): Token[] {
	const out: Token[] = [];
	let i = 0;
	let prevWord = '';
	while (i < src.length) {
		const rest = src.slice(i);
		let matched = false;
		for (const [re, classify] of PATTERNS) {
			const m = re.exec(rest);
			if (!m) continue;
			const text = m[0];
			let cls = classify(text);
			// A bare identifier right after `fn` is a function name.
			if (cls === '' && prevWord === 'fn' && /^[A-Za-z_]/.test(text)) cls = 'tok-fn';
			if (/^[A-Za-z_]/.test(text)) prevWord = text;
			else if (text.trim() !== '') prevWord = '';
			out.push({ text, cls });
			i += text.length;
			matched = true;
			break;
		}
		if (!matched) {
			out.push({ text: src[i], cls: '' });
			i += 1;
		}
	}
	return out;
}
