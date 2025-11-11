const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      ws: true, // Enable WebSocket support for SSE
      timeout: 0, // Disable timeout
      proxyTimeout: 0, // Disable proxy timeout
      onProxyReq: (proxyReq, req, res) => {
        // For SSE endpoints, ensure proper headers
        if (req.url.includes('/progress/')) {
          console.log('📡 SSE request detected for:', req.url);
          proxyReq.setHeader('Accept', 'text/event-stream');
          proxyReq.setHeader('Cache-Control', 'no-cache');
        }
      },
      onProxyRes: (proxyRes, req, res) => {
        // Log SSE response headers for debugging
        if (req.url.includes('/progress/')) {
          console.log('📨 SSE Response headers:', {
            'content-type': proxyRes.headers['content-type'],
            'cache-control': proxyRes.headers['cache-control'],
            'connection': proxyRes.headers['connection']
          });
        }
      },
      onError: (err, req, res) => {
        console.error('❌ Proxy error:', err.message);
      }
    })
  );
};