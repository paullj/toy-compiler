import { syntaxHighlighting } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { toyHighlighter, toyLanguage } from './toy-syntax';

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

export const toyHighlighting = [toyLanguage, syntaxHighlighting(toyHighlighter)];
export const toyExtensions = [...toyHighlighting, theme];
