<script lang="ts">
	import { onMount } from 'svelte';
	import { useResizeObserver } from 'runed';
	import '@xterm/xterm/css/xterm.css';
	import type { Terminal as XTerm } from '@xterm/xterm';
	import type { FitAddon } from '@xterm/addon-fit';
	import { Shell } from '$lib/terminal-shell';

	let el: HTMLDivElement;
	let term = $state<XTerm>();
	let fit: FitAddon | undefined;
	let shell: Shell | undefined;

	useResizeObserver(
		() => el,
		() => {
			try {
				fit?.fit();
			} catch {
				/* container not measurable yet */
			}
		}
	);

	function themeColors() {
		const s = getComputedStyle(document.documentElement);
		const v = (n: string, f: string) => s.getPropertyValue(n).trim() || f;
		return {
			background: v('--code-bg', '#111'),
			foreground: v('--fg', '#eee'),
			cursor: v('--accent', '#e0603a'),
			cursorAccent: v('--code-bg', '#111'),
			selectionBackground: v('--border', '#444'),
			red: v('--accent', '#e0603a')
		};
	}

	/** Called by the page (Run button). */
	export function runFile(path: string) {
		shell?.runFile(path);
	}

	/** Re-apply theme colors when the site theme changes. */
	export function refreshTheme() {
		if (term) term.options.theme = themeColors();
	}

	onMount(() => {
		let disposed = false;
		let sub: { dispose(): void } | undefined;

		(async () => {
			const { Terminal } = await import('@xterm/xterm');
			const { FitAddon } = await import('@xterm/addon-fit');
			if (disposed) return;

			const t = new Terminal({
				convertEol: true,
				cursorBlink: true,
				fontSize: 13,
				fontFamily: getComputedStyle(document.documentElement)
					.getPropertyValue('--font-mono')
					.trim(),
				theme: themeColors(),
				scrollback: 1000
			});
			fit = new FitAddon();
			t.loadAddon(fit);
			t.open(el);
			fit.fit();

			shell = new Shell({ write: (s) => t.write(s) });
			shell.start();
			sub = t.onData((d) => shell!.input(d));
			term = t;
		})();

		return () => {
			disposed = true;
			sub?.dispose();
			term?.dispose();
		};
	});
</script>

<div bind:this={el} class="h-full w-full"></div>
