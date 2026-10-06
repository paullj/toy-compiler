import { expect, it } from 'vitest';
import { highlightLines } from './toy-syntax';

const definitions = (code: string) =>
	highlightLines(code)
		.flat()
		.filter((t) => t.cls === 'tok-fn')
		.map((t) => t.text);

it('marks a declared fn name, not the first parameter of a lambda or a fn type', () => {
	expect(definitions('fn add(a: int, b: int) -> int { a + b }')).toEqual(['add']);
	expect(definitions('xs |> .fold(0, fn(sum, x) { sum + x })')).toEqual([]);
	expect(definitions('fn run(f: fn(T) -> U) {}')).toEqual(['run']);
	expect(definitions('tick := fn() { n = n + 1 }')).toEqual([]);
	expect(definitions('fn main() {\n    fn fact(n: int) -> int { n }\n}')).toEqual(['main', 'fact']);
});
