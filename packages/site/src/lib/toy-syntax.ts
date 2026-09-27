import { StreamLanguage } from '@codemirror/language';
import { highlightCode, tagHighlighter, tags as t } from '@lezer/highlight';

const KEYWORDS = new Set([
	'fn', 'struct', 'enum', 'impl', 'has', 'protocol', 'type', 'match',
	'if', 'else', 'return', 'self', 'mut', 'true', 'false', 'import', 'pub'
]);

const TYPES = new Set([
	'int', 'uint', 'byte', 'char', 'str', 'bool', 'float',
	'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
	'Option', 'Result'
]);

// A small stream tokenizer — good enough for docs and the playground, not a real lexer.
// `depth` counts open brackets so a new line indents one unit per enclosing bracket.
export const toyLanguage = StreamLanguage.define<{ afterFn: boolean; depth: number }>({
	name: 'toy',
	startState: () => ({ afterFn: false, depth: 0 }),
	copyState: (s) => ({ ...s }),
	indent(state, textAfter, cx) {
		const closing = /^[}\])]/.test(textAfter) ? 1 : 0;
		return Math.max(0, state.depth - closing) * cx.unit;
	},
	token(stream, state) {
		if (stream.eatSpace()) return null;
		if (stream.match(/#.*/)) return 'comment';
		if (stream.match(/"(?:\\.|[^"\\])*"/)) return 'string';
		if (stream.match(/'(?:\\.|[^'\\])*'/)) return 'string';
		if (stream.match(/0x[0-9a-fA-F_]+/)) return 'number';
		if (stream.match(/\d[\d_]*(?:\.\d[\d_]*)?/)) return 'number';

		const word = stream.match(/[A-Za-z_][A-Za-z0-9_]*/) as RegExpMatchArray | null;
		if (word) {
			const w = word[0];
			const afterFn = state.afterFn;
			state.afterFn = w === 'fn';
			if (KEYWORDS.has(w)) return 'keyword';
			if (TYPES.has(w)) return 'typeName';
			if (afterFn) return 'variableName.definition';
			return 'variableName';
		}

		const punct = stream.match(/[{}()[\],;:.]|[-+*/%<>=!&|?]+/) as RegExpMatchArray | null;
		if (punct) {
			if (/^[{([]$/.test(punct[0])) state.depth++;
			else if (/^[})\]]$/.test(punct[0])) state.depth = Math.max(0, state.depth - 1);
			return 'punctuation';
		}
		stream.next();
		return null;
	},
	languageData: { commentTokens: { line: '#' }, indentOnInput: /^\s*[}\])]$/ }
});

// Tags to the site's `tok-*` classes (app.css): the editor and the static render share
// one tokenizer and one palette, so a block does not shift when it becomes an editor.
export const toyHighlighter = tagHighlighter([
	{ tag: t.keyword, class: 'tok-kw' },
	{ tag: t.typeName, class: 'tok-type' },
	{ tag: t.number, class: 'tok-num' },
	{ tag: t.string, class: 'tok-str' },
	{ tag: t.comment, class: 'tok-com' },
	{ tag: t.punctuation, class: 'tok-punc' },
	{ tag: t.definition(t.variableName), class: 'tok-fn' }
]);

export type Token = { text: string; cls: string };

// `code` as lines of highlighted runs, for rendering without an editor (SSR, no
// JavaScript). One block per line, as the editor lays them out, so both are the same height.
export function highlightLines(code: string): Token[][] {
	const lines: Token[][] = [[]];
	highlightCode(
		code,
		toyLanguage.parser.parse(code),
		toyHighlighter,
		(text, cls) => lines[lines.length - 1].push({ text, cls }),
		() => lines.push([])
	);
	return lines;
}
