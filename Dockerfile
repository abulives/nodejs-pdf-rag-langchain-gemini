# Build stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .

# Run stage
FROM node:18-slim
WORKDIR /app
COPY --from=builder /app /app
EXPOSE 3000
CMD ["node", "bin/www"]
# This Dockerfile builds a Node.js application in a multi-stage build process.