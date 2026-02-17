# Frontend Dockerfile
FROM node:18-alpine AS builder

ENV NODE_ENV=production

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./

RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi

COPY frontend/src ./src
COPY frontend/public ./public

ENV REACT_APP_API_URL=http://localhost:9000

RUN npm run build

# Serve with nginx
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
