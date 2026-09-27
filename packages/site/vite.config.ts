import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	// The diagnostics reference renders the compiler's `toy explain` pages in place.
	server: { fs: { allow: ['../compiler/src/diagnostics/errors'] } }
});
