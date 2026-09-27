import { marked } from 'marked';

// The compiler's own `toy explain` pages, so the site and the CLI never disagree.
const pages = import.meta.glob('../../../../../compiler/src/diagnostics/errors/*.md', {
	query: '?raw',
	import: 'default',
	eager: true
}) as Record<string, string>;

export function load() {
	const errors = Object.entries(pages)
		.map(([path, md]) => {
			const code = path.slice(path.lastIndexOf('/') + 1, -'.md'.length);
			const [heading, ...rest] = md.trim().split('\n');
			const title = heading.replace(/^#\s*[A-Z]\d{4}:\s*/, '');
			return { code, title, html: marked.parse(rest.join('\n'), { async: false }) };
		})
		.sort((a, b) => a.code.localeCompare(b.code));
	return { errors };
}
