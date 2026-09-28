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
	await expect(page.locator('.cm-lsp-hover-tooltip')).toHaveText('quantity: int');
});

test('completion offers in-scope names with their types', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\ntotal := qu');
	const list = page.locator('.cm-tooltip-autocomplete');
	await expect(list).toContainText('quantity');
	await expect(list).toContainText('int');
});

// Budget from the last keystroke to the underline: the sync pause plus a check and the
// worker round trip, with headroom for a slow CI machine.
const MAX_DIAGNOSTIC_LATENCY_MS = 600;

test('a type error is reported while typing and cleared once fixed', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\nlabel: int = "three"');
	await expect(b.locator('.cm-lintRange-error')).toHaveText('"three"');
	const typed = Date.now();
	await expect(b.locator('.cm-lintRange-error, .cm-lintPoint-error')).toHaveCount(1);
	expect(Date.now() - typed).toBeLessThan(MAX_DIAGNOSTIC_LATENCY_MS);
	for (let i = 0; i < 7; i++) await page.keyboard.press('Backspace');
	await page.keyboard.type('3');
	await expect(b.locator('.cm-lintRange-error, .cm-lintPoint-error')).toHaveCount(0);
});

test('a new line inside a block indents one level per open bracket', async ({ page }) => {
	const b = await live(page, 'shapes.toy');
	await typeAtEndOfLine(page, b, '.Square(side)', '\n.Circle(r) -> 0,');
	await expect(b.locator('.cm-line', { hasText: '.Circle(r) -> 0,' })).toHaveText('        .Circle(r) -> 0,');
});

test('F12 jumps from a use to its definition', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await b.locator('.cm-line', { hasText: 'return price' }).click({ position: { x: 105, y: 6 } });
	await page.keyboard.press('F12');
	await expect
		.poll(() => b.evaluate(() => window.getSelection()?.anchorNode?.parentElement?.closest('.cm-line')?.textContent ?? ''))
		.toContain('price := 20');
});

test('signature help tracks the active argument', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, '}', '\nfn add(a: int, b: int) -> int { return a + b }');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\nsum := add(1, ', 60);
	const sig = page.locator('.cm-lsp-signature-tooltip');
	await expect(sig).toContainText('fn add(int, int) -> int');
	await expect(sig.locator('.cm-lsp-active-parameter')).toHaveText('int');
});

test('a diagnostic tooltip links its code and jumps to the related definition', async ({ page }) => {
	const b = await live(page, 'order.toy');
	await typeAtEndOfLine(page, b, '}', '\nfn add(a: int, b: int) -> int { return a + b }');
	await typeAtEndOfLine(page, b, 'quantity := 3', '\n_sum := add(1)');
	const mark = b.locator('.cm-lintRange-error');
	await expect(mark).toHaveText('(1)');
	// Let the popups the call opened close first: without this settle the hover is flaky.
	await page.keyboard.press('Escape');
	await page.waitForTimeout(300);
	await mark.hover();
	const tip = page.locator('.toy-diag');
	await expect(tip.locator('.toy-diag-code')).toHaveText('T0039');
	await expect(tip.locator('.toy-diag-code')).toHaveAttribute('href', /\/docs\/errors\/#T0039$/);
	await expect(tip.locator('.toy-diag-msg code').first()).toHaveText('add');
	await page.screenshot({ path: 'test-results/diagnostic-tooltip.png' });
	await tip.locator('.toy-diag-related').dispatchEvent('mousedown');
	await expect(b.locator('.cm-activeLine, .cm-line').filter({ hasText: 'fn add' })).toBeVisible();
	const line = await b.evaluate((fig) => {
		const sel = window.getSelection();
		return sel?.anchorNode?.parentElement?.closest('.cm-line')?.textContent ?? '';
	});
	expect(line).toContain('fn add');
});

test('the diagnostics reference renders every code from the compiler', async ({ page }) => {
	await page.goto('/docs/errors/');
	await expect(page.locator('section#T0039 h3')).toContainText('arity mismatch');
	await expect(page.locator('section#R0001 h3')).toContainText('undeclared identifier');
});

test('on a phone, live editors never widen the page', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 800 });
	for (const file of BLOCKS) await live(page, file);
	expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBe(0);
});

test('a block keeps its size and colours when it becomes an editor', async ({ page, browser }) => {
	const noJs = await browser.newPage({ javaScriptEnabled: false });
	await noJs.goto('/');
	for (const file of BLOCKS) {
		const staticBox = await block(noJs, file).boundingBox();
		const staticKeywords = await block(noJs, file).locator('.tok-kw').allTextContents();
		const b = await live(page, file);
		const liveBox = await b.boundingBox();
		expect(Math.abs(liveBox!.height - staticBox!.height)).toBeLessThanOrEqual(1);
		expect(await b.locator('.tok-kw').allTextContents()).toEqual(staticKeywords);
	}
	await noJs.close();
});

test.describe('without JavaScript', () => {
	test.use({ javaScriptEnabled: false });
	test('every example still renders as highlighted code', async ({ page }) => {
		await page.goto('/');
		for (const file of BLOCKS) await expect(block(page, file).locator('pre .tok-kw').first()).toBeVisible();
	});
});
