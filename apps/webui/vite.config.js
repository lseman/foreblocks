import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
      '/nodes': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/execute': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/status': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/logs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/artifact': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/cancel': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
