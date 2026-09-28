<script lang="ts">
	import { resolve } from '$app/paths';
	import squid from '$lib/images/squid-02-stacking.webp';
</script>

<svelte:head>
	<title>Learn — toy</title>
</svelte:head>

<div class="mx-auto flex min-h-screen max-w-6xl items-center justify-center border-x px-6" style="border-color: var(--border)">
	<div class="text-center">
		<img src={squid} alt="The toy squid stacking blocks" class="mx-auto mb-6 w-32 select-none" />
		<p class="font-mono text-[0.7rem] uppercase tracking-[0.2em]" style="color: var(--fg-muted)">Learn</p>
		<h1 class="mt-4 text-3xl">The tour is coming soon.</h1>
		<a href={resolve('/')} class="link mt-6 inline-flex">← Back home</a>
	</div>
</div>
