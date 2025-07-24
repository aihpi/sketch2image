FROM node:18-alpine

WORKDIR /app

# Copy package files and install dependencies first to leverage Docker cache
COPY package*.json ./
RUN npm install

# Copy the rest of the application source code
COPY . .

# Expose the port the React dev server runs on
EXPOSE 3000

# The command to run the app.
# HOST=0.0.0.0 is needed to make the dev server accessible outside the container.
# The REACT_APP_API_URL is passed in from docker-compose.yml.
CMD ["npm", "start"]