FROM node:18-alpine

WORKDIR /app

# Copy package.json and install dependencies
COPY package.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Start React development server on all interfaces
ENV HOST=0.0.0.0
ENV REACT_APP_API_URL=http://localhost:8000/api
CMD ["npm", "start"]
