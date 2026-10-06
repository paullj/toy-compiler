import { Text } from '@codemirror/state';
import { describe, expect, it } from 'vitest';
import type { InlayHint } from 'vscode-languageserver-protocol';
import { placeHints } from './inlay';

const doc = Text.of(['fn main() {', '    bump(count)', '    s := "🙂" + y', '}']);
const at = (line: number, character: number) => ({ line, character });

describe('placeHints', () => {
	it('places a hint before the character its position names', () => {
		const [h] = placeHints(doc, [{ position: at(1, 9), label: 'mut', tooltip: 'mutated by bump', paddingRight: true }]);
		expect(doc.sliceString(h.pos, h.pos + 5)).toBe('count');
		expect(h).toMatchObject({ label: 'mut', tooltip: 'mutated by bump', padLeft: false, padRight: true });
	});

	it('counts characters in UTF-16 code units, as the editor does', () => {
		const [h] = placeHints(doc, [{ position: at(2, 16), label: 'x' }]);
		expect(doc.sliceString(h.pos, h.pos + 1)).toBe('y');
	});

	it('joins label parts and reads a markup tooltip', () => {
		const hint: InlayHint = {
			position: at(0, 10),
			label: [{ value: 'captures ' }, { value: 'count' }],
			tooltip: { kind: 'markdown', value: 'count: shared' }
		};
		expect(placeHints(doc, [hint])).toEqual([
			{ pos: 10, label: 'captures count', tooltip: 'count: shared', padLeft: false, padRight: false }
		]);
	});

	it('drops a position the text does not have', () => {
		const hints = [at(4, 0), at(-1, 0), at(1, 16), at(1, -1)].map((position) => ({ position, label: 'x' }));
		expect(placeHints(doc, hints)).toEqual([]);
		expect(placeHints(doc, [{ position: at(1, 15), label: 'end' }])[0].pos).toBe(doc.line(2).to);
	});
});
