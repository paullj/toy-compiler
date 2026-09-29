<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	// Self-hosted fonts imported as URLs so we control @font-face (font-display)
	// and can preload them — no post-paint swap, no external request.
	import sansRegular from '@fontsource/hanken-grotesk/files/hanken-grotesk-latin-400-normal.woff2?url';
	import sansBold from '@fontsource/hanken-grotesk/files/hanken-grotesk-latin-700-normal.woff2?url';
	import displayRegular from '@fontsource/cherry-bomb-one/files/cherry-bomb-one-latin-400-normal.woff2?url';
	// Ioskeley Mono (SIL OFL 1.1) — self-hosted; see src/lib/fonts/LICENSE.txt.
	import monoRegular from '$lib/fonts/IoskeleyMono-Regular.woff2?url';
	import monoMedium from '$lib/fonts/IoskeleyMono-Medium.woff2?url';
	import monoSemiBold from '$lib/fonts/IoskeleyMono-SemiBold.woff2?url';
	import monoBold from '$lib/fonts/IoskeleyMono-Bold.woff2?url';

	let { children } = $props();

	const faces = `
@font-face{font-family:'Hanken Grotesk';font-style:normal;font-weight:400;font-display:optional;src:url(${sansRegular}) format('woff2')}
@font-face{font-family:'Hanken Grotesk';font-style:normal;font-weight:700;font-display:optional;src:url(${sansBold}) format('woff2')}
@font-face{font-family:'Cherry Bomb One';font-style:normal;font-weight:400;font-display:optional;src:url(${displayRegular}) format('woff2')}
@font-face{font-family:'Ioskeley Mono';font-style:normal;font-weight:400;font-display:optional;src:url(${monoRegular}) format('woff2')}
@font-face{font-family:'Ioskeley Mono';font-style:normal;font-weight:500;font-display:optional;src:url(${monoMedium}) format('woff2')}
@font-face{font-family:'Ioskeley Mono';font-style:normal;font-weight:600;font-display:optional;src:url(${monoSemiBold}) format('woff2')}
@font-face{font-family:'Ioskeley Mono';font-style:normal;font-weight:700;font-display:optional;src:url(${monoBold}) format('woff2')}`;
</script>

<svelte:head>
	<link rel="preload" href={sansRegular} as="font" type="font/woff2" crossorigin="anonymous" />
	<link rel="preload" href={sansBold} as="font" type="font/woff2" crossorigin="anonymous" />
	<link rel="preload" href={displayRegular} as="font" type="font/woff2" crossorigin="anonymous" />
	<link rel="preload" href={monoRegular} as="font" type="font/woff2" crossorigin="anonymous" />
	<!-- eslint-disable-next-line svelte/no-at-html-tags -->
	{@html `<style>${faces}</style>`}
</svelte:head>

<ModeWatcher />
{@render children()}
