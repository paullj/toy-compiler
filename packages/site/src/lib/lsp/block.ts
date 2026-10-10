import { EditorView, keymap } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { indentUnit } from '@codemirror/language';
import { jumpToDefinitionKeymap } from '@codemirror/lsp-client';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { toyHighlighting } from '$lib/toy-language';
import { toyClient } from './client';
import { lspTheme } from './theme';

// Sized to match the static `<pre>` it replaces, so the swap does not shift the page.
const blockTheme = EditorView.theme({
	'&': { color: 'var(--fg)', backgroundColor: 'transparent' },
	'.cm-scroller': { fontFamily: 'var(--font-mono)', fontSize: '0.75rem', lineHeight: '1.625' },
	'.cm-content': { padding: '1rem 0', caretColor: 'var(--accent)' },
	// Pinned like the static render's `h-[1lh]` lines. The mono faces are `font-display:
	// optional`, so on some loads a bold keyword is set in a fallback face whose baseline
	// sits elsewhere, and its line box rounds up a pixel.
	'.cm-line': { padding: '0 1rem', height: '1lh' },
	'&.cm-focused': { outline: 'none' },
	'.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--accent)' },
	'&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': {
		backgroundColor: 'color-mix(in srgb, var(--accent) 22%, transparent)'
	}
});

export function mountBlock(parent: HTMLElement, doc: string, uri: string): EditorView {
	return new EditorView({
		parent,
		state: EditorState.create({
			doc,
			extensions: [
				history(),
				closeBrackets(),
				indentUnit.of('    '),
				keymap.of([...closeBracketsKeymap, ...jumpToDefinitionKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
				...toyHighlighting,
				blockTheme,
				lspTheme,
				toyClient().plugin(uri, 'toy')
			]
		})
	});
}
