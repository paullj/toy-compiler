<script lang="ts">
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { useDebounce, useEventListener, watch } from 'runed';
	import { mode } from 'mode-watcher';
	import { PaneGroup, Pane, PaneResizer } from 'paneforge';
	import Editor from '$lib/components/Editor.svelte';
	import Terminal from '$lib/components/Terminal.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
	import * as vfs from '$lib/vfs';

	const SEED = [
		{ path: 'main.toy', content: `fn main() {\n    print("hello, world\\n")\n}\n` },
		{
			path: 'src/math.toy',
			content: `fn main() -> int {\n    price := 20        # inferred int\n    quantity := 3\n    return price * quantity\n}\n`
		},
		{
			path: 'src/shapes.toy',
			content: `enum Shape { Circle(int), Square(int) }\n\nfn area(shape: Shape) -> int {\n    match shape {\n        .Circle(r) -> 3 * r * r,\n        .Square(side) -> side * side,\n    }\n}\n`
		}
	];

	let tree = $state<vfs.Entry[]>([]);
	let expanded = new SvelteSet<string>();
	let selected = $state<string | null>(null);
	let openPath = $state<string | null>(null);
	let code = $state('');
	let loaded = '';

	let creating = $state<{ kind: 'file' | 'dir'; parent: string } | null>(null);
	let newName = $state('');

	let termRef = $state<Terminal>();

	type Row = { entry: vfs.Entry; depth: number };
	function flatten(entries: vfs.Entry[], depth = 0, acc: Row[] = []): Row[] {
		for (const e of entries) {
			acc.push({ entry: e, depth });
			if (e.kind === 'dir' && expanded.has(e.path)) flatten(e.children, depth + 1, acc);
		}
		return acc;
	}
	const rows = $derived(flatten(tree));

	async function refresh() {
		tree = await vfs.readTree();
	}

	async function openFile(path: string) {
		selected = path;
		openPath = path;
		const c = await vfs.readFile(path);
		loaded = c;
		code = c;
	}

	function clickEntry(e: vfs.Entry) {
		selected = e.path;
		if (e.kind === 'dir') {
			if (expanded.has(e.path)) expanded.delete(e.path);
			else expanded.add(e.path);
		} else {
			openFile(e.path);
		}
	}

	// Persist edits back to the VFS (debounced via runed).
	const saveFile = useDebounce(async (p: string, c: string) => {
		await vfs.writeFile(p, c);
		loaded = c;
	}, 350);
	$effect(() => {
		const c = code;
		const p = openPath;
		if (!p || c === loaded) return;
		saveFile(p, c);
	});

	function targetDir(): string {
		if (!selected) return '';
		const node = rows.find((r) => r.entry.path === selected)?.entry;
		if (node?.kind === 'dir') return node.path;
		if (selected.includes('/')) return selected.split('/').slice(0, -1).join('/');
		return '';
	}

	function startCreate(kind: 'file' | 'dir') {
		creating = { kind, parent: targetDir() };
		newName = '';
	}

	async function confirmCreate() {
		if (!creating) return;
		const name = newName.trim().replace(/\/+/g, '');
		if (!name) {
			creating = null;
			return;
		}
		const path = creating.parent ? `${creating.parent}/${name}` : name;
		if (creating.parent) expanded.add(creating.parent);
		if (creating.kind === 'file') {
			await vfs.writeFile(path, '');
			await refresh();
			await openFile(path);
		} else {
			await vfs.createDir(path);
			expanded.add(path);
			await refresh();
		}
		creating = null;
	}

	async function del(path: string, ev: MouseEvent) {
		ev.stopPropagation();
		await vfs.remove(path);
		if (openPath && (openPath === path || openPath.startsWith(path + '/'))) {
			openPath = null;
			code = '';
			loaded = '';
		}
		await refresh();
	}

	async function run() {
		if (!openPath) return;
		await vfs.writeFile(openPath, code);
		loaded = code;
		termRef?.runFile(openPath);
	}

	onMount(() => {
		(async () => {
			await vfs.seedIfEmpty(SEED);
			await refresh();
			expanded.add('src');
			if (await vfs.exists('main.toy')) await openFile('main.toy');
		})();
	});

	// Re-tint the terminal when the resolved theme changes (after the class is applied).
	watch(
		() => mode.current,
		() => {
			requestAnimationFrame(() => termRef?.refreshTheme());
		}
	);

	// Run the open file with ⌘/Ctrl+↵.
	useEventListener(
		() => window,
		'keydown',
		(e) => {
			if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
				e.preventDefault();
				run();
			}
		}
	);
</script>

<svelte:head>
	<title>Playground — toy</title>
	<meta name="description" content="A preview of the toy language playground." />
</svelte:head>

<div class="mx-auto flex h-screen max-w-6xl flex-col border-x" style="border-color: var(--border)">
	<!-- Header -->
	<header class="flex h-14 shrink-0 items-center gap-4 border-b px-5" style="border-color: var(--border)">
		<a href={resolve('/')} class="flex items-center gap-2 font-mono text-sm font-medium">
			<span class="text-base leading-none">🧸</span> toy
		</a>
		<span class="font-mono text-[0.7rem] uppercase tracking-[0.2em]" style="color: var(--fg-muted)">
			Playground
		</span>
		<span
			class="rounded border px-1.5 py-0.5 font-mono text-[0.6rem] uppercase tracking-widest"
			style="border-color: var(--border); color: var(--fg-muted)"
			title="Programs aren't really compiled yet — output is emulated."
		>
			preview
		</span>
		<div class="ml-auto flex items-center gap-2 text-[0.72rem]">
			<span class="font-mono text-[0.6rem] uppercase tracking-widest" style="color: var(--fg-muted)">
				Theme
			</span>
			<ThemeToggle class="text-[0.72rem]" />
		</div>
	</header>

	<div class="min-h-0 flex-1">
		<PaneGroup direction="horizontal" class="flex h-full">
			<!-- File explorer -->
			<Pane defaultSize={22} minSize={14} class="flex min-h-0 min-w-0 flex-col border-r" style="border-color: var(--border)">
			<div
				class="flex shrink-0 items-center gap-2 border-b px-3 py-2 font-mono text-[0.62rem] uppercase tracking-widest"
				style="border-color: var(--border); color: var(--fg-muted)"
			>
				Files
				<span class="ml-auto flex gap-1">
					<button type="button" title="New file" onclick={() => startCreate('file')} class="px-1 transition-colors hover:text-[var(--fg)]">＋file</button>
					<button type="button" title="New folder" onclick={() => startCreate('dir')} class="px-1 transition-colors hover:text-[var(--fg)]">＋dir</button>
				</span>
			</div>

			<div class="flex-1 overflow-auto py-1 text-[0.8rem]">
				{#if creating}
					<div class="flex items-center gap-1 px-3 py-1 font-mono text-[0.75rem]" style="color: var(--fg-muted)">
						<span>{creating.kind === 'dir' ? '📁' : '📄'}</span>
						<!-- svelte-ignore a11y_autofocus -->
						<input
							bind:value={newName}
							autofocus
							placeholder={creating.parent ? `${creating.parent}/…` : '…'}
							onkeydown={(e) => {
								if (e.key === 'Enter') confirmCreate();
								else if (e.key === 'Escape') (creating = null);
							}}
							onblur={confirmCreate}
							class="w-full bg-transparent outline-none"
							style="color: var(--fg)"
						/>
					</div>
				{/if}

				{#each rows as { entry, depth } (entry.path)}
					<div
						class="group flex items-center font-mono transition-colors"
						style={selected === entry.path
							? 'color: var(--fg); background: var(--code-bg)'
							: 'color: var(--fg-muted)'}
					>
						<button
							type="button"
							onclick={() => clickEntry(entry)}
							class="flex min-w-0 flex-1 items-center gap-1.5 py-0.5 text-left transition-colors hover:text-[var(--fg)]"
							style="padding-left: {0.75 + depth * 0.85}rem"
						>
							<span class="w-3 shrink-0 text-[0.7rem]">
								{entry.kind === 'dir' ? (expanded.has(entry.path) ? '▾' : '▸') : ''}
							</span>
							<span class="truncate">{entry.name}</span>
						</button>
						<button
							type="button"
							title="Delete"
							onclick={(e) => del(entry.path, e)}
							class="shrink-0 px-2 opacity-0 transition-opacity hover:text-[var(--accent)] group-hover:opacity-70"
						>✕</button>
					</div>
				{/each}
			</div>
			</Pane>

			<PaneResizer class="pf-resizer" />

			<!-- Editor + terminal -->
			<Pane defaultSize={78} minSize={30} class="min-h-0 min-w-0">
				<PaneGroup direction="vertical" class="flex h-full flex-col">
					<Pane defaultSize={64} minSize={20} class="flex min-h-0 min-w-0 flex-col">
			<div
				class="flex h-10 shrink-0 items-center gap-3 border-b px-4"
				style="border-color: var(--border)"
			>
				<span class="font-mono text-[0.75rem]" style="color: var(--fg-muted)">
					{openPath ?? 'no file open'}
				</span>
				<button
					type="button"
					onclick={run}
					disabled={!openPath}
					class="btn btn-fill ml-auto text-[0.78rem] disabled:opacity-40"
					style="padding: 0.3rem 0.85rem"
				>
					Run <span class="font-mono text-[0.85em] opacity-70">⌘↵</span>
				</button>
			</div>

			<div class="min-h-0 min-w-0 flex-1 overflow-hidden" style="background: var(--code-bg)">
				{#if openPath}
					<Editor bind:value={code} />
				{:else}
					<div class="grid h-full place-items-center text-[0.85rem]" style="color: var(--fg-muted)">
						Select a file to start editing.
					</div>
				{/if}
			</div>

					</Pane>

					<PaneResizer class="pf-resizer" />

					<Pane defaultSize={36} minSize={12} class="min-h-0 min-w-0 overflow-hidden border-t" style="border-color: var(--border); background: var(--code-bg)">
						<Terminal bind:this={termRef} />
					</Pane>
				</PaneGroup>
			</Pane>
		</PaneGroup>
	</div>
</div>
