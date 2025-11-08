# Tweet Search Frontend

Modern, responsive frontend for semantic tweet search built with Next.js 14, React, and Tailwind CSS.

## Features

- Clean, responsive UI
- Real-time search results
- Similarity scores for each result
- Tweet engagement stats (retweets, likes)
- Fast and lightweight

## Development

### Prerequisites

- Node.js 18+ or Bun
- Backend API running (Hetzner server or local)

### Setup

1. Install dependencies:
```bash
npm install
# or
bun install
```

2. Create `.env.local` file:
```bash
cp .env.example .env.local
```

3. Update `NEXT_PUBLIC_API_URL` in `.env.local` to point to your backend:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
# or your Hetzner server IP
NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP
```

4. Run development server:
```bash
npm run dev
# or
bun dev
```

5. Open [http://localhost:3000](http://localhost:3000)

## Production Deployment

### Option 1: Vercel (Recommended)

1. Push your code to GitHub
2. Import project to [Vercel](https://vercel.com)
3. Set environment variable:
   - `NEXT_PUBLIC_API_URL` = `http://YOUR_HETZNER_IP` or `https://your-domain.com`
4. Deploy!

### Option 2: Deploy on Hetzner Server

1. Build the production app:
```bash
npm run build
```

2. Copy `frontend/.next` and `frontend/node_modules` to your Hetzner server

3. On the server, run:
```bash
cd frontend
npm start
```

4. Configure Nginx to serve the frontend on port 80/443

### Option 3: Static Export

For pure static hosting (Netlify, GitHub Pages, etc.):

1. Update `next.config.js`:
```js
const nextConfig = {
  output: 'export',
  // ...
}
```

2. Build:
```bash
npm run build
```

3. Deploy the `out/` directory

## Environment Variables

- `NEXT_PUBLIC_API_URL` - Backend API URL (required)

## Tech Stack

- **Next.js 14** - React framework
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Vercel** - Deployment (recommended)

## API Integration

The frontend expects the backend API at `/search` endpoint with the following format:

**Request:**
```
GET /search?query=bitcoin&limit=20
```

**Response:**
```json
{
  "query": "bitcoin",
  "results": [
    {
      "tweet_id": 123456,
      "full_text": "...",
      "username": "...",
      "created_at": "2024-01-01T00:00:00Z",
      "similarity": 0.95,
      "retweet_count": 100,
      "favorite_count": 500
    }
  ],
  "search_time_ms": 250,
  "database_size": 6400000
}
```
