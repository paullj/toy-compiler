<script lang="ts">
	import { onMount } from 'svelte';
	import { tokenize } from '$lib/highlight';

	let { code, filename }: { code: string; filename?: string } = $props();

	const source = $derived(code.trim());
	const tokens = $derived(tokenize(source));

	let host: HTMLDivElement;
	let live = $state(false);

	// The static render is the first paint (and the no-JS page); a block becomes a live
	// editor only once it scrolls near the viewport, so the editor and language server
	// load only for readers who reach the examples.
	onMount(() => {
		let view: { destroy(): void } | undefined;
		const io = new IntersectionObserver(
			async (entries) => {
				if (!entries.some((e) => e.isIntersecting)) return;
				io.disconnect();
				const { mountBlock } = await import('$lib/lsp/block');
				view = mountBlock(host, source, `file:///examples/${filename ?? 'block.toy'}`);
				live = true;
			},
			{ rootMargin: '200px' }
		);
		io.observe(host);
		return () => {
			io.disconnect();
			view?.destroy();
		};
	});
</script>

<figure
	class="my-4 rounded-xl border"
	style="border-color: var(--border); background: var(--code-bg)"
>
	{#if filename}
		<figcaption
			class="flex items-center gap-2 border-b px-4 py-2 font-mono text-[0.7rem]"
			style="border-color: var(--border); color: var(--fg-muted)"
		>
			<span class="h-2.5 w-2.5 rounded-full border" style="border-color: var(--fg-muted)"></span>
			{filename}
			{#if live}<span class="ml-auto">editable</span>{/if}
		</figcaption>
	{/if}
	<div bind:this={host} class="overflow-x-auto">
		{#if !live}
			<pre class="p-4 font-mono text-[0.75rem] leading-relaxed"><code
					>{#each tokens as t}<span class={t.cls}>{t.text}</span>{/each}</code
				></pre>
		{/if}
	</div>
</figure>
