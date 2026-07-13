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
    // Node definition colors are data-driven, so keep only the combinations
    // actually emitted by nodeDefinitions.js.
    'from-blue-600',
    'to-blue-700',
    'from-green-600',
    'to-green-700',
    'from-amber-600',
    'to-orange-700',

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
