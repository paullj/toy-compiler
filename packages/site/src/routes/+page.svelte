<script lang="ts">
	import { useEventListener } from 'runed';
	import Code from '$lib/components/Code.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';

	const nav = [
		{ id: 'introduction', label: 'Introduction' },
		{ id: 'features', label: 'Features' },
		{ id: 'install', label: 'Install' },
		{ id: 'learn', label: 'Learn' }
	];

	const secondary = [
		{ href: '/tour/', label: 'Tour →' },
		{ href: '/playground/', label: 'Playground →' },
		{ href: '/docs/', label: 'Reference →' },
		{ href: 'https://github.com/paullj/toy-compiler', label: 'Source →' }
	];

	const featureItems = [
		{
			title: 'Type inference',
			blurb: 'Bind a value with <code class="code">:=</code> and its type is inferred. Annotate the boundaries and let the body follow.',
			file: 'order.toy',
			code: `fn main() -> int {
    price := 20        # inferred int
    quantity := 3
    return price * quantity
}`
		},
		{
			title: 'Pattern matching',
			blurb: 'Enums hold one of several variants. <code class="code">match</code> destructures them, checked for exhaustiveness at compile time.',
			file: 'shapes.toy',
			code: `enum Shape { Circle(int), Square(int) }

fn area(shape: Shape) -> int {
    match shape {
        .Circle(r) -> 3 * r * r,
        .Square(side) -> side * side,
    }
}`
		},
		{
			title: 'Generics',
			blurb: 'Generic structs and functions are monomorphized. <code class="code">Pair[int, int]</code> reifies to an ordinary struct — no boxing.',
			file: 'pair.toy',
			code: `struct Pair[A, B] { left: A, right: B }

fn main() -> int {
    point := Pair[int, int] { left: 3, right: 39 }
    return point.left + point.right
}`
		},
		{
			title: 'Protocols',
			blurb: 'A protocol is a set of method signatures. Conform a type with <code class="code">impl Square has Area</code> and call its methods directly.',
			file: 'area.toy',
			code: `protocol Area {
    fn area(self) -> int
}

struct Square { side: int }

impl Square has Area {
    fn area(self) -> int { self.side * self.side }
}`
		},
		{
			title: 'No null',
			blurb: '<code class="code">Option[T]</code> and <code class="code">Result[T, E]</code> are ordinary enums. The postfix <code class="code">?</code> unwraps a value or returns early on the empty case.',
			file: 'total.toy',
			code: `fn total(a: Option[int], b: Option[int]) -> Option[int] {
    x := a?
    y := b?
    Option.some(x + y)
}`
		}
	];

	const callouts = [
		{ href: '/tour/', kind: 'Tour', label: 'Take the tour →' },
		{ href: '/playground/', kind: 'Playground', label: 'Try it in the browser →' },
		{ href: '/docs/', kind: 'Reference', label: 'Read the docs →' },
		{ href: 'https://github.com/paullj/toy-compiler', kind: 'Source', label: 'View on GitHub →' }
	];

	// --- Scrollspy ---
	// Active = the last section whose heading has reached the top of the viewport.
	// The line sits just below where a clicked heading lands (its scroll-margin),
	// so short and tall sections are both detected correctly.
	let active = $state(nav[0].id);

	function updateActive() {
		const mark = 80;
		let cur = nav[0].id;
		for (const s of nav) {
			const el = document.getElementById(s.id);
			if (el && el.getBoundingClientRect().top <= mark) cur = s.id;
		}
		active = cur;
	}

	useEventListener(() => window, 'scroll', updateActive, { passive: true });
	useEventListener(() => window, 'resize', updateActive);
	$effect(updateActive);
</script>

<svelte:head>
	<title>toy. a small toy language</title>
	<meta name="description" content="toy is a general purpose language that is statically typed and compiled to native code." />
</svelte:head>

<div class="mx-auto min-h-screen max-w-6xl border-x" style="border-color: var(--border)">
	<div class="grid md:grid-cols-[240px_1fr]">
		<!-- Sidebar -->
		<aside
			class="sticky top-0 hidden h-screen flex-col border-r px-7 pb-8 pt-28 md:flex"
			style="border-color: var(--border)"
		>
			<a href="/" class="mb-8 inline-flex items-center gap-2 text-xl leading-none">🧸</a>

			<nav class="flex flex-col gap-1 text-[0.82rem]">
				{#each nav as s}
					<a
						href="#{s.id}"
						class="side-link py-0.5"
						class:active={active === s.id}
					>
						{s.label}
					</a>
				{/each}
			</nav>

			<div class="mt-6 flex flex-col gap-1 border-t pt-5 text-[0.82rem]" style="border-color: var(--border)">
				{#each secondary as l}
					<a href={l.href} class="side-link py-0.5">{l.label}</a>
				{/each}
			</div>

			<!-- Theme -->
			<div class="mt-auto flex items-center gap-2 pt-8 text-[0.82rem]">
				<span class="font-mono text-[0.6rem] uppercase tracking-widest" style="color: var(--fg-muted)">
					Theme
				</span>
				<ThemeToggle class="text-[0.82rem]" />
			</div>
		</aside>

		<!-- Content -->
		<main class="min-w-0 px-6 pb-16 pt-28 sm:px-10 md:px-12 md:pt-[10.5rem]">
			<!-- 01 Introduction -->
			<section id="introduction" class="grid scroll-mt-8 grid-cols-[2.75rem_1fr] pb-24">
				<div class="mt-2 font-mono text-[0.7rem]" style="color: var(--fg-muted)">01</div>
				<div>
					<h1 class="max-w-[16ch] text-3xl font-bold leading-[1.05] tracking-tight md:text-4xl">
						<a href="#introduction" class="transition-colors hover:text-[var(--accent)]">
							A small toy language.
						</a>
					</h1>
					<p class="mt-5 max-w-[48ch] text-base leading-snug" style="color: var(--fg-muted)">
						toy is a general purpose language which is statically typed, compiled, and garbage
						collected.
					</p>
					<p class="mt-4 max-w-[48ch] text-base leading-snug" style="color: var(--fg-muted)">
It aims to be ergonomic and performant but the main goal is for me to learn about language design and compiler development.
										</p>
				</div>
			</section>

			<!-- 02 Features -->
			<section id="features" class="grid scroll-mt-8 grid-cols-[2.75rem_1fr] pb-24">
				<div class="mt-1.5 font-mono text-[0.7rem]" style="color: var(--fg-muted)">02</div>
				<div>
					<h2 class="text-xl font-bold tracking-tight">
						<a href="#features" class="transition-colors hover:text-[var(--accent)]">Features</a>
					</h2>
					<p class="mt-3 max-w-[48ch] text-sm leading-snug" style="color: var(--fg-muted)">
						A modern type system over value types, with the ergonomics you expect and none of the
						ceremony.
					</p>

					<div class="mt-8 border-t" style="border-color: var(--border)">
						{#each featureItems as f}
							<div class="border-b py-7" style="border-color: var(--border)">
								<h3 class="text-sm font-medium tracking-tight">{f.title}</h3>
								<p class="mt-1.5 max-w-[54ch] text-sm leading-snug" style="color: var(--fg-muted)">
									{@html f.blurb}
								</p>
								<div class="mt-4 max-w-2xl">
									<Code code={f.code} filename={f.file} />
								</div>
							</div>
						{/each}
					</div>
				</div>
			</section>

			<!-- 03 Install -->
			<section id="install" class="grid scroll-mt-8 grid-cols-[2.75rem_1fr] pb-24">
				<div class="mt-1.5 font-mono text-[0.7rem]" style="color: var(--fg-muted)">03</div>
				<div>
					<h2 class="text-xl font-bold tracking-tight">
						<a href="#install" class="transition-colors hover:text-[var(--accent)]">Install</a>
					</h2>
					<p class="mt-3 max-w-[48ch] text-sm leading-snug" style="color: var(--fg-muted)">
						toy currently has to be built from
						<a href="https://github.com/paullj/toy-compiler" class="link">source</a>.
					</p>
				</div>
			</section>

			<!-- 04 Learn -->
			<section id="learn" class="grid scroll-mt-8 grid-cols-[2.75rem_1fr] pb-24">
				<div class="mt-1.5 font-mono text-[0.7rem]" style="color: var(--fg-muted)">04</div>
				<div>
					<h2 class="text-xl font-bold tracking-tight">
						<a href="#learn" class="transition-colors hover:text-[var(--accent)]">Learn</a>
					</h2>
					<p class="mt-3 max-w-[48ch] text-sm leading-snug" style="color: var(--fg-muted)">
						Ready to go deeper? Walk the language tour, browse the reference, or read the source.
					</p>

					<div
						class="mt-7 grid gap-px border sm:grid-cols-2"
						style="border-color: var(--border); background: var(--border)"
					>
						{#each callouts as c}
							<a
								href={c.href}
								class="group flex flex-col gap-2 p-6 transition-colors hover:bg-[var(--code-bg)]"
								style="background: var(--bg)"
							>
								<span class="font-mono text-[0.6rem] uppercase tracking-widest" style="color: var(--fg-muted)">
									{c.kind}
								</span>
								<span class="text-sm font-medium tracking-tight transition-colors group-hover:text-[var(--accent)]">
									{c.label}
								</span>
							</a>
						{/each}
					</div>
				</div>
			</section>
			<!-- Trailing whitespace so the last section can scroll up to the mark. -->
			<div aria-hidden="true" class="h-[60vh]"></div>

			<footer
				class="flex items-center justify-between border-t pt-6 font-mono text-[0.7rem]"
				style="border-color: var(--border); color: var(--fg-muted)"
			>
				<span>🧸 toy</span>
				<span>a small toy language</span>
			</footer>
		</main>
	</div>
</div>
