# Use Node.js LTS version
FROM node:20-alpine AS build

# Set work directory
WORKDIR /app

# Copy package files first for caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend source code
COPY . .

# Expose Vite's default port
EXPOSE 5173

# Run the dev server (you could switch to a prod build + serve if needed)
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "5173"]

