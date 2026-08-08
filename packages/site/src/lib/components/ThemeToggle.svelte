<script lang="ts">
	import { userPrefersMode, setMode } from 'mode-watcher';

	let { class: klass = '' }: { class?: string } = $props();

	const order = ['system', 'light', 'dark'] as const;

	function cycle() {
		const i = order.indexOf(userPrefersMode.current as (typeof order)[number]);
		setMode(order[(i + 1) % order.length]);
	}
</script>

<button
	type="button"
	onclick={cycle}
	title="Theme: click to cycle system / light / dark"
	class="theme-btn font-mono leading-none lowercase {klass}"
>
	{userPrefersMode.current}
</button>
