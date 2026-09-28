import { LSPPlugin, type LSPClient, type LSPClientExtension } from '@codemirror/lsp-client';
import { setDiagnostics, type Diagnostic } from '@codemirror/lint';
import { ViewPlugin, type EditorView, type ViewUpdate } from '@codemirror/view';
import { asset } from '$app/paths';
import type * as lsp from 'vscode-languageserver-protocol';

type Severity = Diagnostic['severity'];

const SEVERITY: Record<lsp.DiagnosticSeverity, Severity> = { 1: 'error', 2: 'warning', 3: 'info', 4: 'hint' };

// The server republishes after every sync. Re-dispatching an identical set would close an
// open diagnostic tooltip under the reader's cursor, so an unchanged set is dropped.
const shown = new WeakMap<EditorView, string>();

// A check costs a few milliseconds, so edits are pushed after a short pause (the stock
// extension waits 500ms): diagnostics then keep up with typing.
const SYNC_DELAY_MS = 120;

const autoSync = ViewPlugin.fromClass(
	class {
		pending: ReturnType<typeof setTimeout> | undefined;
		update(u: ViewUpdate) {
			if (!u.docChanged) return;
			clearTimeout(this.pending);
			this.pending = setTimeout(() => LSPPlugin.get(u.view)?.client.sync(), SYNC_DELAY_MS);
		}
		destroy() {
			clearTimeout(this.pending);
		}
	}
);

// In place of the client's `serverDiagnostics`: the tooltip leads with the code (linked to
// its reference entry), sets quoted names in code, and turns each related location into a
// button that jumps to it.
export function toyDiagnostics(): LSPClientExtension {
	return {
		clientCapabilities: { textDocument: { publishDiagnostics: { versionSupport: true, relatedInformation: true } } },
		notificationHandlers: { 'textDocument/publishDiagnostics': publishDiagnostics },
		editorExtension: autoSync
	};
}

function publishDiagnostics(client: LSPClient, params: lsp.PublishDiagnosticsParams) {
	const file = client.workspace.getFile(params.uri);
	if (!file || (params.version != null && params.version !== file.version)) return false;
	const view = file.getView();
	const plugin = view && LSPPlugin.get(view);
	if (!view || !plugin) return false;

	// Clamped: a position past the synced text (a trailing line with no newline) would make
	// `mapPos` throw, taking the whole publish down with it.
	const toPos = (p: lsp.Position) =>
		plugin.unsyncedChanges.mapPos(Math.min(plugin.fromPosition(p, plugin.syncedDoc), plugin.syncedDoc.length));
	const diagnostics: Diagnostic[] = params.diagnostics.map((d) => {
		const severity = SEVERITY[d.severity ?? 1];
		const message = typeof d.message === 'string' ? d.message : d.message.value;
		const related = (d.relatedInformation ?? [])
			.filter((r) => r.location.uri === params.uri)
			.map((r) => ({ label: r.message, pos: toPos(r.location.range.start) }));
		return {
			from: toPos(d.range.start),
			to: toPos(d.range.end),
			severity,
			message,
			renderMessage: (v) => render(v, severity, d.code, message, related)
		};
	});
	// Marks the editor as checked at least once, for tests that assert on "no diagnostics".
	view.dom.dataset.checked = String(params.version ?? '');
	const key = JSON.stringify(diagnostics.map((d) => [d.from, d.to, d.severity, d.message]));
	if (shown.get(view) === key) return true;
	shown.set(view, key);
	view.dispatch(setDiagnostics(view.state, diagnostics));
	return true;
}

function render(view: EditorView, severity: Severity, code: lsp.Diagnostic['code'], message: string, related: { label: string; pos: number }[]) {
	const root = el('div', 'toy-diag');
	const head = root.appendChild(el('div', 'toy-diag-head'));
	head.appendChild(el('span', `toy-diag-sev toy-diag-sev-${severity}`, severity));
	if (code != null) {
		const link = head.appendChild(el('a', 'toy-diag-code', String(code))) as HTMLAnchorElement;
		link.href = asset(`/docs/errors/#${code}`);
		link.target = '_blank';
		link.title = `What ${code} means and how to fix it`;
	}
	root.appendChild(messageNode(message));
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
