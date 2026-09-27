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
	'.cm-diagnostic': { padding: '0.5rem 0.7rem', borderLeftWidth: '3px', maxWidth: '34rem' },
	'.toy-diag': { display: 'flex', flexDirection: 'column', gap: '0.3rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' },
	'.toy-diag-head': { display: 'flex', alignItems: 'baseline', gap: '0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.68rem' },
	'.toy-diag-sev': { textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600' },
	'.toy-diag-sev-error': { color: 'var(--accent)' },
	'.toy-diag-sev-warning': { color: '#c89a2c' },
	'.toy-diag-sev-info, .toy-diag-sev-hint': { color: 'var(--fg-muted)' },
	'.toy-diag-code': { color: 'var(--fg-muted)', textDecoration: 'underline dotted', textUnderlineOffset: '3px' },
	'.toy-diag-code:hover': { color: 'var(--accent)' },
	'.toy-diag-msg': { lineHeight: '1.45' },
	'.toy-diag-msg code': {
		fontFamily: 'var(--font-mono)',
		fontSize: '0.92em',
		padding: '0 0.25em',
		borderRadius: '0.25rem',
		background: 'var(--code-bg)',
		border: '1px solid var(--border)'
	},
	'.toy-diag-related': {
		alignSelf: 'flex-start',
		font: 'inherit',
		fontFamily: 'var(--font-mono)',
		fontSize: '0.7rem',
		color: 'var(--fg-muted)',
		background: 'none',
		border: 'none',
		padding: '0',
		cursor: 'pointer'
	},
	'.toy-diag-related:hover': { color: 'var(--accent)' },
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
