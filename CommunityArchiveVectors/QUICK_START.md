# Quick Start Guide

Get the Community Archive running in 15 minutes (without semantic search) or 1 hour (with semantic search).

## 🚀 Fast Track (No Semantic Search)

This setup gives you 95% of functionality - network visualization, topics, tweets - everything except semantic search.

### Prerequisites
- Supabase account (free tier works)
- Vercel account (free tier works)
- Node.js 18+

### Steps

1. **Setup Supabase**
   ```bash
   # 1. Create project at supabase.com
   # 2. Run this SQL in the SQL editor:
   ```
   ```sql
   CREATE TABLE account (
     account_id BIGINT PRIMARY KEY,
     username TEXT NOT NULL,
     account_display_name TEXT
   );

   CREATE TABLE tweets (
     tweet_id BIGINT PRIMARY KEY,
     account_id BIGINT REFERENCES account(account_id),
     full_text TEXT NOT NULL,
     created_at TIMESTAMP,
     retweet_count INTEGER DEFAULT 0,
     favorite_count INTEGER DEFAULT 0,
     reply_to_tweet_id BIGINT
   );

   CREATE INDEX idx_tweets_account_id ON tweets(account_id);
   CREATE INDEX idx_account_username ON account(username);
   ```

   Import your tweet data via Supabase dashboard or API.

2. **Configure Frontend**
   ```bash
   cd frontend
   cp .env.example .env.local
   # Edit .env.local with your Supabase credentials
   ```

3. **Deploy**
   ```bash
   npm install
   npm run build
   npx vercel --prod
   ```

4. **Add Environment Variables in Vercel**
   - Go to Vercel dashboard → Your Project → Settings → Environment Variables
   - Add:
     - NEXT_PUBLIC_SUPABASE_URL = Your Supabase project URL
     - NEXT_PUBLIC_SUPABASE_ANON_KEY = Your Supabase anon key
   - Redeploy

**Done!** Your site should be live at your Vercel URL.

---

## 🔍 Full Setup (With Semantic Search)

Adds powerful semantic search functionality over millions of tweets.

### Additional Prerequisites
- A VPS server with 32GB RAM (Vultr, DigitalOcean, AWS)
- Modal account for data processing
- ~100GB disk space on server
- Python 3.11+

### Steps

1. **Complete Fast Track setup above** ✓

2. **Provision a Server**
   ```bash
   # Example: Vultr High Performance
   # 8 vCPUs, 32GB RAM, 100GB SSD
   # Ubuntu 22.04 LTS
   # Note the IP address
   ```

3. **Transfer Embeddings**
   ```bash
   # From your local machine with Modal access
   modal setup
   python3 direct_transfer_to_vultr.py
   # Follow prompts, enter your server IP
   ```

4. **Setup Server**

   SSH into your server and run:
   ```bash
   # Install dependencies
   apt update
   apt install -y python3.11 python3-pip
   pip3 install fastapi uvicorn numpy sentence-transformers torch psycopg2-binary

   # Download the server script
   mkdir -p /root/tweet-search
   cd /root/tweet-search

   # Copy server.py from DEPLOYMENT.md Part 4, Step 4
   # Edit DB_CONFIG with your Supabase credentials
   nano server.py
   ```

5. **Create Systemd Service**
   ```bash
   # Copy service file from DEPLOYMENT.md Part 4, Step 6
   nano /etc/systemd/system/tweet-search.service

   systemctl daemon-reload
   systemctl enable tweet-search
   systemctl start tweet-search
   ```

6. **Configure Firewall**
   ```bash
   ufw allow 80/tcp
   ufw allow 22/tcp
   ufw enable
   ```

7. **Update Frontend**

   In Vercel environment variables, add:
   ```
   BACKEND_URL=http://YOUR_SERVER_IP
   ```

   Redeploy frontend.

8. **Test**
   ```bash
   curl "http://YOUR_SERVER_IP/search?query=climate%20change&limit=5"
   ```

**Done!** Your semantic search should now work.

---

## 📊 What You Get

### Fast Track Setup
- ✅ Network constellation visualization
- ✅ Community topics and clusters
- ✅ Tweet browsing and viewing
- ✅ User network exploration
- ✅ Timeline scrubbing through years
- ✅ Community lineage tracking
- ❌ Semantic search

### Full Setup
- ✅ Everything from Fast Track
- ✅ Semantic search over 6.4M tweets
- ✅ Natural language queries
- ✅ Similarity-based tweet discovery

---

## 💰 Cost

### Fast Track
- **$0-25/month**
  - Vercel: Free (Hobby) or $20 (Pro)
  - Supabase: Free (hobby) or $25 (Pro) depending on data size

### Full Setup
- **$65-125/month**
  - Frontend + Database: $0-45/month
  - VPS Server: $40-80/month (32GB RAM)

---

## 🐛 Common Issues

### "No tweets found"
→ Check Supabase credentials are set in Vercel environment variables

### Build fails
→ Run `rm -rf .next && npm run build`

### Server out of memory
→ Upgrade to 32GB+ RAM or reduce embedding batch sizes

### Search not working
→ Check `systemctl status tweet-search` and firewall settings

---

## 📚 Full Documentation

See DEPLOYMENT.md for:
- Complete architecture details
- Database schema
- Server configuration
- Performance tuning
- Monitoring and backups
- Troubleshooting

---

## 🆘 Need Help?

1. Check logs:
   - Frontend: Vercel dashboard → Logs
   - Backend: `journalctl -u tweet-search -f`
   - Database: Supabase dashboard → Logs

2. Common fixes:
   - Restart services: `systemctl restart tweet-search`
   - Check env vars: Verify all credentials are set
   - Test connectivity: `curl http://YOUR_SERVER_IP`

3. Still stuck? Check GitHub issues or create a new one with logs.
