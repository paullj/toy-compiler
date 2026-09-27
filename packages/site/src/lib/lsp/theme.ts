import { EditorView } from '@codemirror/view';

// Tooltips, the completion list, and lint marks in the site's own palette.
export const lspTheme = EditorView.theme({
	'.cm-tooltip': {
		backgroundColor: 'var(--bg-elev)',
		color: 'var(--fg)',
		border: '1px solid var(--border)',
		borderRadius: '0.5rem',
		boxShadow: '0 6px 24px -8px rgb(0 0 0 / 0.18)',
		fontFamily: 'var(--font-mono)',
		fontSize: '0.75rem',
		overflow: 'hidden'
	},
	'.cm-tooltip-hover, .cm-lsp-signature-tooltip': { padding: '0.4rem 0.6rem', maxWidth: '36rem' },
	'.cm-tooltip-hover code, .cm-lsp-signature-tooltip code': { fontFamily: 'var(--font-mono)' },
	'.cm-tooltip-hover pre': { margin: '0' },
	'.cm-lsp-active-parameter': { color: 'var(--accent)', fontWeight: '600' },
	'.cm-tooltip.cm-tooltip-autocomplete > ul': { fontFamily: 'var(--font-mono)', maxHeight: '14rem' },
	'.cm-tooltip.cm-tooltip-autocomplete > ul > li': { padding: '0.15rem 0.6rem' },
	'.cm-tooltip-autocomplete ul li[aria-selected]': {
		backgroundColor: 'color-mix(in srgb, var(--accent) 16%, transparent)',
		color: 'var(--fg)'
	},
	'.cm-completionDetail': { color: 'var(--fg-muted)', fontStyle: 'normal', marginLeft: '1em' },
	'.cm-completionMatchedText': { textDecoration: 'none', color: 'var(--accent)' },
	'.cm-diagnostic': { padding: '0.4rem 0.6rem', borderLeftWidth: '3px' },
	'.cm-diagnostic-error': { borderLeftColor: 'var(--accent)' },
	'.cm-diagnostic-warning': { borderLeftColor: '#c89a2c' },
	'.cm-lintRange-error': {
		backgroundImage: 'none',
		textDecoration: 'underline wavy var(--accent)',
		textUnderlineOffset: '3px'
	},
	'.cm-lintRange-warning': {
		backgroundImage: 'none',
		textDecoration: 'underline wavy #c89a2c',
		textUnderlineOffset: '3px'
	}
});
