import { defineConfig } from 'vitepress'
import { readFileSync } from 'fs'
import { resolve } from 'path'

const pyproject = readFileSync(resolve(__dirname, '../../pyproject.toml'), 'utf-8')
const version = pyproject.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? 'unknown'

const docsSidebar = [
    {
        text: 'Start Here',
        items: [
            { text: 'Home', link: '/' },
            { text: 'Overview', link: '/overview' },
            { text: 'Getting Started', link: '/getting-started' },
        ],
    },
    {
        text: 'Tutorials',
        collapsed: false,
        items: [
            { text: 'Train A Direct Model', link: '/tutorials/train-direct-model' },
            { text: 'Run A DARTS Search', link: '/tutorials/darts-multifidelity-search' },
            { text: 'Generate Synthetic Series', link: '/tutorials/generate-synthetic-series' },
            { text: 'Optimize With BOHB', link: '/tutorials/optimize-with-bohb' },
        ],
    },
    {
        text: 'Guides',
        collapsed: false,
        items: [
            { text: 'Preprocessor', link: '/preprocessor' },
            { text: 'Custom Blocks', link: '/custom_blocks' },
            { text: 'Transformer', link: '/transformer' },
            { text: 'Mixture Of Experts', link: '/moe' },
            { text: 'Hybrid Mamba', link: '/hybrid-mamba' },
            { text: 'DARTS', link: '/darts' },
            { text: 'Evaluation & Metrics', link: '/evaluation' },
            { text: 'Uncertainty Quantification', link: '/uncertainty' },
            { text: 'Web UI', link: '/webui' },
            { text: 'Troubleshooting', link: '/troubleshooting' },
        ],
    },
    {
        text: 'Foretools',
        collapsed: true,
        items: [
            { text: 'Foretools Overview', link: '/foretools/index' },
            { text: 'Time Series Generator', link: '/foretools/tsgen' },
            { text: 'BOHB Search', link: '/foretools/bohb' },
            { text: 'VMD Decomposition', link: '/foretools/vmd' },
            { text: 'AutoDA Augmentation', link: '/foretools/tsaug' },
            { text: 'Feature Engineering', link: '/foretools/feature-engineering' },
        ],
    },
    {
        text: 'Architecture',
        collapsed: true,
        items: [
            { text: 'System Overview', link: '/architecture/system-overview' },
            { text: 'Forecasting Pipeline', link: '/architecture/forecasting-pipeline' },
            { text: 'DARTS Search Pipeline', link: '/architecture/darts-pipeline' },
        ],
    },
    {
        text: 'Reference',
        collapsed: true,
        items: [
            { text: 'Public API', link: '/reference/public-api' },
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'Repository Map', link: '/reference/repository-map' },
        ],
    },
    {
        text: 'Contributing',
        collapsed: true,
        items: [
            { text: 'Documentation Workflow', link: '/contributing/docs-workflow' },
        ],
    },
    {
        text: 'Release Notes',
        collapsed: true,
        items: [
            { text: 'Changelog', link: '/changelog' },
        ],
    },
]

export default defineConfig({
    title: 'foreBlocks',
    description: 'Modular time-series forecasting, preprocessing, and architecture search for PyTorch',
    base: '/docs/',
    define: {
        __FOREBLOCKS_VERSION__: JSON.stringify(version),
    },
    transformPageData(pageData) {
        pageData.frontmatter.foreBlocksVersion = version
    },
    appearance: 'force-dark',
    head: [
        ['link', { rel: 'icon', href: '/docs/logo.svg' }],
    ],
    themeConfig: {
        logo: '/logo.svg',
        nav: [],
        sidebar: docsSidebar,
        socialLinks: [
            { icon: 'github', link: 'https://github.com/lseman/foreblocks' },
        ],
        editLink: {
            pattern: 'https://github.com/lseman/foreblocks/edit/main/docs/:path',
            text: 'Edit this page on GitHub',
        },
        footer: {
            message: 'MIT License',
            copyright: 'Copyright © 2026 foreBlocks',
        },
    },
    markdown: {
        config: (md) => {
            const defaultFence = md.renderer.rules.fence

            md.renderer.rules.fence = (tokens, idx, options, env, self) => {
                const token = tokens[idx]

                if (token.info.trim() === 'mermaid') {
                    const encodedSource = md.utils.escapeHtml(encodeURIComponent(token.content))
                    return `<div class="mermaid" data-source="${encodedSource}"></div>`
                }

                return defaultFence
                    ? defaultFence(tokens, idx, options, env, self)
                    : self.renderToken(tokens, idx, options)
            }
        },
    },
})
