import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, '.', '');
    return {
        base: './',
        plugins: [react()],
        server: {
            host: '0.0.0.0',
            port: 5175,
            proxy: {
                '/api': {
                    target: env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
                    changeOrigin: true,
                },
            },
        },
    };
});
