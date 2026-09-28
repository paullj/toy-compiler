<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>Learn — toy</title>
</svelte:head>

<div class="mx-auto flex min-h-screen max-w-6xl items-center justify-center border-x px-6" style="border-color: var(--border)">
	<div class="text-center">
		<p class="font-mono text-[0.7rem] uppercase tracking-[0.2em]" style="color: var(--fg-muted)">Learn</p>
		<h1 class="mt-4 text-2xl font-semibold tracking-tight">The tour is coming soon.</h1>
		<a href={resolve('/')} class="link mt-6 inline-flex">← Back home</a>
	</div>
</div>
