import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({ fallback: '404.html' }),
		// GitHub Pages serves the site under /<repo>; CI sets this for the Pages build only.
		paths: { base: process.env.BASE_PATH ?? '' }
	}
};

export default config;
