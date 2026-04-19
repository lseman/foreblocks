import DefaultTheme from 'vitepress/theme'
import { inBrowser, onContentUpdated, useData } from 'vitepress'
import mermaid from 'mermaid'
import { nextTick, watch } from 'vue'

import '../../stylesheets/extra.css'

async function renderMermaidDiagrams(isDark) {
    if (!inBrowser) {
        return
    }

    const diagrams = document.querySelectorAll('.vp-doc .mermaid')

    if (!diagrams.length) {
        return
    }

    mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: isDark ? 'dark' : 'default',
    })

    for (const diagram of diagrams) {
        const source = diagram.dataset.source
            ? decodeURIComponent(diagram.dataset.source)
            : diagram.textContent ?? ''

        diagram.removeAttribute('data-processed')
        diagram.textContent = source
    }

    await mermaid.run({ nodes: diagrams })
}

export default {
    extends: DefaultTheme,
    setup() {
        const { isDark } = useData()

        const updateMermaid = () => {
            void nextTick(() => renderMermaidDiagrams(isDark.value))
        }

        onContentUpdated(updateMermaid)
        watch(isDark, updateMermaid)
    },
}
