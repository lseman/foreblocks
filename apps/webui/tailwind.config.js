/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
    './src/features/**/*.{js,ts,jsx,tsx}',
  ],

  safelist: [
    // ─────────────────────────────────────────────
    // Gradient directions
    // ─────────────────────────────────────────────
    'bg-gradient-to-r',
    'bg-gradient-to-l',
    'bg-gradient-to-t',
    'bg-gradient-to-b',
    'bg-gradient-to-tr',
    'bg-gradient-to-tl',
    'bg-gradient-to-br',
    'bg-gradient-to-bl',

    // ─────────────────────────────────────────────
    // Color scales (from/to/bg)
    // ─────────────────────────────────────────────
    {
      pattern:
        /(from|to|via|bg|text|border|ring)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(50|100|200|300|400|500|600|700|800|900)/,
    },

    // ─────────────────────────────────────────────
    // Opacity variants (for modals, overlays, etc.)
    // ─────────────────────────────────────────────
    {
      pattern:
        /bg-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(700|800|900)\/(10|20|30|40|50|60|70|80|90|95)/,
    },

    // Explicit fallbacks often used in themes
    'bg-slate-700/60',
    'bg-slate-900/80',
    'bg-slate-800/70',
  ],

  theme: {
    extend: {
      backdropBlur: {
        xs: '2px',
      },
      zIndex: {
        max: '9999',
      },
      boxShadow: {
        modal: '0 10px 40px rgba(0, 0, 0, 0.45)',
      },
      transitionTimingFunction: {
        'in-out-soft': 'cubic-bezier(0.45, 0, 0.55, 1)',
      },
    },
  },

  plugins: [],
};
