# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files and install
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Express Proxy Server
FROM node:18-alpine AS production

WORKDIR /app

# Copy proxy server files
COPY proxy-server/package*.json ./
RUN npm install

# Copy proxy server code
COPY proxy-server/server.js ./

# Copy React build from previous stage
COPY --from=frontend-builder /app/frontend/build ./build

EXPOSE 3000

CMD ["npm", "start"]