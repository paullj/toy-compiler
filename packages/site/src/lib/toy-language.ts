import { StreamLanguage, HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';

const KEYWORDS = new Set([
	'fn', 'struct', 'enum', 'impl', 'has', 'protocol', 'type', 'match',
	'if', 'else', 'return', 'self', 'mut', 'true', 'false', 'import', 'pub'
]);

const TYPES = new Set([
	'int', 'uint', 'byte', 'char', 'str', 'bool', 'float',
	'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
	'Option', 'Result'
]);

// A small stream tokenizer — good enough for the playground, not a real lexer.
const toyLanguage = StreamLanguage.define<{ afterFn: boolean }>({
	startState: () => ({ afterFn: false }),
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

		stream.next();
		return null;
	},
	languageData: { commentTokens: { line: '#' } }
});

const highlight = HighlightStyle.define([
	{ tag: t.keyword, color: 'var(--tok-kw)', fontWeight: '600' },
	{ tag: t.typeName, color: 'var(--tok-type)' },
	{ tag: t.number, color: 'var(--tok-num)' },
	{ tag: t.string, color: 'var(--tok-str)' },
	{ tag: t.comment, color: 'var(--fg-muted)', fontStyle: 'italic' },
	{ tag: t.definition(t.variableName), color: 'var(--tok-fn)' }
]);

const theme = EditorView.theme({
	'&': { color: 'var(--fg)', backgroundColor: 'transparent', height: '100%' },
	'.cm-scroller': {
		fontFamily: 'var(--font-mono)',
		fontSize: '0.8125rem',
		lineHeight: '1.6'
	},
	'.cm-content': { padding: '1rem 0' },
	'.cm-gutters': {
		backgroundColor: 'transparent',
		color: 'color-mix(in srgb, var(--fg-muted) 55%, transparent)',
		border: 'none',
		paddingRight: '0.5rem'
	},
	'.cm-lineNumbers .cm-gutterElement': { padding: '0 0.5rem 0 1rem' },
	'&.cm-focused': { outline: 'none' },
	'.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--accent)' },
	'.cm-activeLine': { backgroundColor: 'color-mix(in srgb, var(--fg) 3.5%, transparent)' },
	'.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--fg-muted)' },
	'&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': {
		backgroundColor: 'color-mix(in srgb, var(--accent) 22%, transparent)'
	}
});

export const toyExtensions = [toyLanguage, syntaxHighlighting(highlight), theme];
