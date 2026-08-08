<script lang="ts">
	import { onMount } from 'svelte';
	import { EditorView, keymap, lineNumbers, highlightActiveLine } from '@codemirror/view';
	import { EditorState } from '@codemirror/state';
	import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
	import { toyExtensions } from '$lib/toy-language';

	let { value = $bindable('') }: { value?: string } = $props();

	let el: HTMLDivElement;
	let view = $state<EditorView>();

	onMount(() => {
		view = new EditorView({
			parent: el,
			state: EditorState.create({
				doc: value,
				extensions: [
					lineNumbers(),
					highlightActiveLine(),
					history(),
					keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
					...toyExtensions,
					EditorView.updateListener.of((u) => {
						if (u.docChanged) value = u.state.doc.toString();
					})
				]
			})
		});
		return () => view?.destroy();
	});

	// External value changes (e.g. loading an example) → replace the document.
	$effect(() => {
		if (view && value !== view.state.doc.toString()) {
			view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
		}
	});
</script>

<div bind:this={el} class="h-full overflow-auto"></div>
