import { LSPPlugin, type LSPClient } from '@codemirror/lsp-client';
import { setDiagnostics, type Diagnostic } from '@codemirror/lint';
import type { EditorView } from '@codemirror/view';
import { asset } from '$app/paths';

type LspPosition = { line: number; character: number };
type LspRange = { start: LspPosition; end: LspPosition };
type LspDiagnostic = {
	range: LspRange;
	severity?: number;
	code?: string;
	message: string;
	relatedInformation?: { location: { uri: string; range: LspRange }; message: string }[];
};

const SEVERITY = ['error', 'error', 'warning', 'info', 'hint'] as const;

// The server republishes after every sync. Re-dispatching an identical set would close an
// open diagnostic tooltip under the reader's cursor, so an unchanged set is dropped.
const shown = new WeakMap<EditorView, string>();

// Replaces the client's plain-text rendering: the tooltip leads with the code (linked to
// its reference entry), sets quoted names and types in code, and turns each related
// location into a button that jumps to it.
export function publishDiagnostics(client: LSPClient, params: { uri: string; version?: number; diagnostics: LspDiagnostic[] }) {
	const file = client.workspace.getFile(params.uri);
	if (!file || (params.version != null && params.version !== file.version)) return false;
	const view = file.getView();
	const plugin = view && LSPPlugin.get(view);
	if (!view || !plugin) return false;

	const toPos = (p: LspPosition) => plugin.unsyncedChanges.mapPos(plugin.fromPosition(p, plugin.syncedDoc));
	const diagnostics: Diagnostic[] = params.diagnostics.map((d) => {
		const severity = SEVERITY[d.severity ?? 1];
		const related = (d.relatedInformation ?? [])
			.filter((r) => r.location.uri === params.uri)
			.map((r) => ({ label: r.message, pos: toPos(r.location.range.start) }));
		return {
			from: toPos(d.range.start),
			to: toPos(d.range.end),
			severity: severity === 'hint' ? 'hint' : severity,
			message: d.message,
			renderMessage: (v) => render(v, severity, d, related)
		};
	});
	const key = JSON.stringify(diagnostics.map((d) => [d.from, d.to, d.severity, d.message]));
	if (shown.get(view) === key) return true;
	shown.set(view, key);
	view.dispatch(setDiagnostics(view.state, diagnostics));
	return true;
}

function render(view: EditorView, severity: string, d: LspDiagnostic, related: { label: string; pos: number }[]) {
	const root = el('div', 'toy-diag');
	const head = root.appendChild(el('div', 'toy-diag-head'));
	head.appendChild(el('span', `toy-diag-sev toy-diag-sev-${severity}`, severity));
	if (d.code) {
		const code = head.appendChild(el('a', 'toy-diag-code', d.code)) as HTMLAnchorElement;
		code.href = asset(`/docs/errors/#${d.code}`);
		code.target = '_blank';
		code.title = `What ${d.code} means and how to fix it`;
	}
	root.appendChild(messageNode(d.message));
	for (const r of related) {
		const line = view.state.doc.lineAt(r.pos).number;
		const btn = root.appendChild(el('button', 'toy-diag-related', `↳ ${r.label} · line ${line}`));
		btn.addEventListener('mousedown', (e) => {
			e.preventDefault();
			view.dispatch({ selection: { anchor: r.pos }, scrollIntoView: true });
			view.focus();
		});
	}
	return root;
}

function messageNode(message: string) {
	const p = el('div', 'toy-diag-msg');
	const parts = message.split(/('[^']+')/);
	for (const part of parts) {
		if (part.length > 2 && part.startsWith("'") && part.endsWith("'")) p.appendChild(el('code', '', part.slice(1, -1)));
		else if (part) p.appendChild(document.createTextNode(part));
	}
	return p;
}

function el(tag: string, cls: string, text?: string) {
	const n = document.createElement(tag);
	if (cls) n.className = cls;
	if (text != null) n.textContent = text;
	return n;
}
