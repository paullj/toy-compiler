import { expect, test, type Page } from '@playwright/test';

const BLOCKS = ['order.toy', 'shapes.toy', 'pair.toy', 'area.toy', 'total.toy'];

function block(page: Page, file: string) {
	return page.locator('figure', { has: page.locator('figcaption', { hasText: file }) });
}

async function live(page: Page, file: string) {
	const b = block(page, file);
	await b.scrollIntoViewIfNeeded();
	await expect(b.locator('.cm-editor')).toBeVisible();
	return b;
}

// `delay` types at human speed: the client drops an in-flight signature request when the
// cursor moves, so zero-delay typing would outrun every reply.
async function typeAtEndOfLine(page: Page, b: ReturnType<typeof block>, lineText: string, text: string, delay = 0) {
	await b.locator('.cm-line', { hasText: lineText }).first().click();
	await page.keyboard.press('End');
	await page.keyboard.type(text, { delay });
}

test.beforeEach(async ({ page }) => {
	await page.goto('/');
});

test('every example becomes an editor and checks clean', async ({ page }) => {
	for (const file of BLOCKS) {
		const b = await live(page, file);
		await expect(b.locator('.cm-lintRange, .cm-lintPoint')).toHaveCount(0);
	}
	// Given time for the server to publish, a clean example must stay clean.
	await page.waitForTimeout(500);
	await expect(page.locator('.cm-lintRange, .cm-lintPoint')).toHaveCount(0);
});

test('hovering a binding shows its inferred type', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await b.locator('.cm-line', { hasText: 'quantity := 3' }).hover({ position: { x: 50, y: 6 } });
	await expect(page.locator('.cm-tooltip-hover')).toHaveText('int');
});

test('completion offers in-scope names with their types', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\ntotal := qu');
	const list = page.locator('.cm-tooltip-autocomplete');
	await expect(list).toContainText('quantity');
	await expect(list).toContainText('int');
});

test('a type error is reported while typing and cleared once fixed', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\nlabel: int = "three"');
	await expect(b.locator('.cm-lintRange-error, .cm-lintPoint-error')).toHaveCount(1);
	for (let i = 0; i < 7; i++) await page.keyboard.press('Backspace');
	await page.keyboard.type('3');
	await expect(b.locator('.cm-lintRange-error, .cm-lintPoint-error')).toHaveCount(0);
});

test('signature help tracks the active argument', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, '}', '\nfn add(a: int, b: int) -> int { return a + b }');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\nsum := add(1, ', 60);
	const sig = page.locator('.cm-lsp-signature-tooltip');
	await expect(sig).toContainText('fn add(int, int) -> int');
	await expect(sig.locator('.cm-lsp-active-parameter')).toHaveText('int');
});
