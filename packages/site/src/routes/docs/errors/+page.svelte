<script lang="ts">
	import { resolve } from '$app/paths';
	let { data } = $props();

	const bands = [
		{ prefix: 'P', label: 'Parse' },
		{ prefix: 'R', label: 'Resolve' },
		{ prefix: 'T', label: 'Type' },
		{ prefix: 'W', label: 'Warnings' }
	];
</script>

<svelte:head>
	<title>Diagnostics — toy</title>
	<meta name="description" content="Every error and warning code the toy compiler reports, and how to fix it." />
</svelte:head>

<div class="mx-auto min-h-screen max-w-6xl border-x px-6 pb-24 pt-20 sm:px-10 md:px-12" style="border-color: var(--border)">
	<a href={resolve('/')} class="link text-[0.82rem]">← toy</a>
	<h1 class="mt-8 text-4xl">Diagnostics</h1>
	<p class="mt-3 max-w-[56ch] text-base leading-snug" style="color: var(--fg-muted)">
		Every coded error and warning the compiler reports. The same text ships in the compiler as
		<code class="code">toy explain CODE</code>.
	</p>

	{#each bands as band}
		<h2 class="mt-14 border-b pb-2 font-mono text-[0.7rem] uppercase tracking-widest" style="border-color: var(--border); color: var(--fg-muted)">
			{band.label}
		</h2>
		{#each data.errors.filter((e) => e.code.startsWith(band.prefix)) as e (e.code)}
			<section id={e.code} class="scroll-mt-8 border-b py-7" style="border-color: var(--border)">
				<h3 class="flex items-baseline gap-3 text-sm font-medium tracking-tight">
					<a href="#{e.code}" class="font-mono text-[0.8rem] transition-colors hover:text-[var(--accent)]">{e.code}</a>
					<span>{e.title}</span>
				</h3>
				<div class="prose-toy mt-3 max-w-[64ch] text-sm leading-relaxed">{@html e.html}</div>
			</section>
		{/each}
	{/each}
</div>

<style>
	.prose-toy :global(p) {
		margin: 0.6rem 0;
		color: var(--fg-muted);
	}
	.prose-toy :global(ul) {
		margin: 0.6rem 0;
		padding-left: 1.2rem;
		list-style: disc;
		color: var(--fg-muted);
	}
	.prose-toy :global(pre) {
		margin: 0.8rem 0;
		padding: 0.9rem 1rem;
		overflow-x: auto;
		border: 1px solid var(--border);
		border-radius: 0.6rem;
		background: var(--code-bg);
		font-size: 0.75rem;
	}
	.prose-toy :global(:not(pre) > code) {
		font-family: var(--font-mono);
		font-size: 0.82em;
		padding: 0.1em 0.35em;
		border-radius: 0.3rem;
		background: var(--code-bg);
		border: 1px solid var(--border);
		color: var(--fg);
	}
</style>
