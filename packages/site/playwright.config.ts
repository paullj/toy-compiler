import { defineConfig } from '@playwright/test';

export default defineConfig({
	testDir: 'e2e',
	webServer: { command: 'pnpm build && pnpm preview --port 4173 --strictPort', port: 4173, timeout: 120_000 },
	use: { baseURL: 'http://localhost:4173' }
});
