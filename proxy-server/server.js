const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';

console.log('🔥 NEW SSE-ENABLED SERVER VERSION LOADED 🔥');
console.log('Starting proxy server...');
console.log('Backend URL:', BACKEND_URL);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', backend: BACKEND_URL });
});

// Single proxy configuration for ALL API endpoints including SSE
app.use('/api', createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  ws: true, // Enable WebSocket support
  timeout: 0, // Disable timeout completely
  proxyTimeout: 0, // Disable proxy timeout completely
  onProxyReq: (proxyReq, req, res) => {
    console.log('🔄 Proxying:', req.method, req.url);
    
    // For progress endpoints, ensure SSE headers
    if (req.url.includes('/progress/')) {
      console.log('📡 SSE request detected for:', req.url);
      proxyReq.setHeader('Accept', 'text/event-stream');
      proxyReq.setHeader('Cache-Control', 'no-cache');
    }
  },
  onProxyRes: (proxyRes, req, res) => {
    // Log response headers for debugging
    if (req.url.includes('/progress/')) {
      console.log('📨 SSE Response headers:', {
        'content-type': proxyRes.headers['content-type'],
        'cache-control': proxyRes.headers['cache-control'],
        'connection': proxyRes.headers['connection']
      });
    }
    
    // Don't modify SSE responses - let them pass through unchanged
    if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
      console.log('✅ SSE stream detected - passing through unchanged');
    }
  },
  onError: (err, req, res) => {
    console.error('❌ Proxy error:', err.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Proxy error', details: err.message });
    }
  }
}));

// Serve static React app
app.use(express.static(path.join(__dirname, 'build')));

// Handle React routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Proxy server running on port ${PORT}`);
  console.log(`📡 SSE support enabled for /api/progress/*`);
});