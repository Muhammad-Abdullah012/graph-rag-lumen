# Frontend Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./

RUN npm ci

COPY frontend/src ./src
COPY frontend/public ./public

ENV REACT_APP_API_URL=http://localhost:8080/api

RUN npm run build

# Serve with nginx
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
