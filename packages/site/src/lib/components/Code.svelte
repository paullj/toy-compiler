<script lang="ts">
	import { tokenize } from '$lib/highlight';

	let { code, filename }: { code: string; filename?: string } = $props();

	const tokens = $derived(tokenize(code.trim()));
</script>

<figure
	class="my-4 overflow-hidden rounded-xl border"
	style="border-color: var(--border); background: var(--code-bg)"
>
	{#if filename}
		<figcaption
			class="flex items-center gap-2 border-b px-4 py-2 font-mono text-[0.7rem]"
			style="border-color: var(--border); color: var(--fg-muted)"
		>
			<span class="h-2.5 w-2.5 rounded-full border" style="border-color: var(--fg-muted)"></span>
			{filename}
		</figcaption>
	{/if}
	<pre class="overflow-x-auto p-4 font-mono text-[0.75rem] leading-relaxed"><code
			>{#each tokens as t}<span class={t.cls}>{t.text}</span>{/each}</code
		></pre>
</figure>
