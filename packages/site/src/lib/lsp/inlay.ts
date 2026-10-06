import { LSPPlugin, type LSPClient, type LSPClientExtension } from '@codemirror/lsp-client';
import { StateEffect, StateField, type Text } from '@codemirror/state';
import { Decoration, EditorView, WidgetType, type DecorationSet } from '@codemirror/view';
import type * as lsp from 'vscode-languageserver-protocol';

export type Hint = { pos: number; label: string; tooltip: string; padLeft: boolean; padRight: boolean };

// Hints placed in `doc`, the text the server answered for. A position past that text is
// dropped rather than clamped: clamped, it would sit beside the wrong token.
export function placeHints(doc: Text, hints: lsp.InlayHint[]): Hint[] {
	const placed: Hint[] = [];
	for (const h of hints) {
		const { line, character } = h.position;
		if (line < 0 || line >= doc.lines) continue;
		const at = doc.line(line + 1);
		if (character < 0 || character > at.length) continue;
		placed.push({
			pos: at.from + character,
			label: typeof h.label === 'string' ? h.label : h.label.map((part) => part.value).join(''),
			tooltip: typeof h.tooltip === 'string' ? h.tooltip : (h.tooltip?.value ?? ''),
			padLeft: !!h.paddingLeft,
			padRight: !!h.paddingRight
		});
	}
	return placed;
}

class HintWidget extends WidgetType {
	constructor(readonly hint: Hint) {
		super();
	}

	eq(other: HintWidget) {
		const a = this.hint;
		const b = other.hint;
		return a.label === b.label && a.tooltip === b.tooltip && a.padLeft === b.padLeft && a.padRight === b.padRight;
	}

	toDOM() {
		const span = document.createElement('span');
		span.className = 'toy-inlay';
		span.classList.toggle('toy-inlay-pad-left', this.hint.padLeft);
		span.classList.toggle('toy-inlay-pad-right', this.hint.padRight);
		span.textContent = this.hint.label;
		if (this.hint.tooltip) span.title = this.hint.tooltip;
		return span;
	}

	// A click on a hint places the cursor as a click on the code beside it would.
	ignoreEvent() {
		return false;
	}
}

const setHints = StateEffect.define<Hint[]>();

const hintField = StateField.define<DecorationSet>({
	create: () => Decoration.none,
	update(hints, tr) {
		for (const e of tr.effects) {
			if (!e.is(setHints)) continue;
			// Side -1: a hint belongs to the text after it, so the cursor and anything typed at
			// its position land between the two.
			const ranges = e.value.map((h) => Decoration.widget({ widget: new HintWidget(h), side: -1 }).range(h.pos));
			return Decoration.set(ranges, true);
		}
		return hints.map(tr.changes);
	},
	provide: (f) => EditorView.decorations.from(f)
});

function requestHints(client: LSPClient, uri: string, view: EditorView) {
	const plugin = LSPPlugin.get(view);
	if (!plugin) return;
	const doc = plugin.syncedDoc;
	const range = { start: { line: 0, character: 0 }, end: plugin.toPosition(doc.length, doc) };
	// The mapping carries the answer over any edits made while it was in flight.
	client
		.withMapping((mapping) =>
			client
				.request<lsp.InlayHintParams, lsp.InlayHint[] | null>('textDocument/inlayHint', { textDocument: { uri }, range })
				.then((hints) => {
					// The editor may have been destroyed while the request was in flight.
					if (LSPPlugin.get(view) !== plugin) return;
					const placed = placeHints(doc, hints ?? []).map((h) => ({ ...h, pos: mapping.mapPos(uri, h.pos) }));
					view.dispatch({ effects: setHints.of(placed) });
					// Marks the editor as hinted at least once, for tests that measure it after.
					view.dom.dataset.hinted = '';
				})
		)
		// A missed refresh only leaves the last hints in place until the next check.
		.catch(() => {});
}

// The server sends no `workspace/inlayHint/refresh`. A diagnostics publish marks a fresh
// check of a document instead (after it opens, after each debounced sync, after a restarted
// server re-opens it), so hints are asked for then. Returning false passes the publish on
// to the diagnostics handler.
export function toyInlayHints(): LSPClientExtension {
	return {
		clientCapabilities: { textDocument: { inlayHint: {} } },
		notificationHandlers: {
			'textDocument/publishDiagnostics': (client, params: lsp.PublishDiagnosticsParams) => {
				const file = client.workspace.getFile(params.uri);
				const view = file?.getView();
				if (file && view && (params.version == null || params.version === file.version)) requestHints(client, params.uri, view);
				return false;
			}
		},
		editorExtension: hintField
	};
}
