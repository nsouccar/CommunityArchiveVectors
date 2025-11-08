# Complete Deployment Guide - Tweet Search

This guide walks you through deploying your tweet search system from Modal to Hetzner with a web UI.

## Architecture Overview

```
┌─────────────────────┐
│   Frontend (Vercel) │  ← Next.js React app
│   or Static Host    │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  Hetzner Server     │  ← FastAPI backend + CoreNN database
│  (€44/month)        │
│  64GB RAM           │
│  - FastAPI API      │
│  - CoreNN DB (40GB) │
│  - Nginx (reverse   │
│    proxy)           │
└─────────────────────┘
```

## Part 1: Get Database from Modal

### Step 1: Download Database

The database backup script is already running. Check its progress:

```bash
# Check the backup progress (from your local machine)
modal volume ls tweet-vectors-volume
```

Once the backup is complete (you'll see `corenn_backup.tar.gz` file), download it:

```bash
# Download the 40GB backup from Modal to your local machine
modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz
```

This will take a while (~1-2 hours depending on your internet speed for 40GB).

## Part 2: Setup Hetzner Server

### Step 1: Order Server

1. Go to [Hetzner Robot](https://robot.hetzner.com/order) or [Server Auction](https://www.hetzner.com/sb)

2. Recommended specs:
   - **EX44** or similar
   - 64GB RAM (minimum)
   - 2x SSD
   - 1 Gbit/s connection
   - Cost: ~€44/month

3. Note your server IP address after activation

### Step 2: Initial Server Setup

SSH into your new server:

```bash
ssh root@YOUR_SERVER_IP
```

Run initial setup:

```bash
# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y python3.11 python3.11-venv python3-pip git nginx certbot python3-certbot-nginx

# Create deploy user
adduser deploy
usermod -aG sudo deploy

# Switch to deploy user
su - deploy
```

### Step 3: Setup Application Directory

```bash
cd ~
mkdir tweet-search
cd tweet-search
```

## Part 3: Transfer and Setup Backend

### Step 1: Transfer Files from Your Local Machine

From your local machine (where this repo is):

```bash
# Transfer backend files
scp hetzner_backend.py deploy@YOUR_SERVER_IP:~/tweet-search/
scp requirements_hetzner.txt deploy@YOUR_SERVER_IP:~/tweet-search/
scp deploy_hetzner.sh deploy@YOUR_SERVER_IP:~/tweet-search/

# Transfer database (this will take a while - 40GB)
scp corenn_backup.tar.gz deploy@YOUR_SERVER_IP:~/tweet-search/
```

### Step 2: Extract Database on Hetzner

SSH back into your Hetzner server:

```bash
ssh deploy@YOUR_SERVER_IP
cd ~/tweet-search

# Extract database (creates corenn_db/ and metadata.pkl)
tar -xzf corenn_backup.tar.gz

# Verify extraction
ls -lh
# You should see:
# - corenn_db/ (directory, ~39GB)
# - metadata.pkl (~1.5GB)
```

### Step 3: Deploy Backend

```bash
cd ~/tweet-search

# Run deployment script
chmod +x deploy_hetzner.sh
./deploy_hetzner.sh
```

### Step 4: Configure Environment

Create `.env` file:

```bash
nano ~/tweet-search/.env
```

Add your Voyage API key:

```
VOYAGE_API_KEY=your_actual_voyage_api_key_here
PORT=8000
```

Save and exit (Ctrl+X, Y, Enter).

### Step 5: Test Backend Locally

```bash
cd ~/tweet-search
source venv/bin/activate
python hetzner_backend.py
```

Wait for the database to load (will show "SERVER READY!" message). Then in another terminal:

```bash
curl "http://localhost:8000/search?query=bitcoin&limit=5"
```

If you see results, the backend is working! Press Ctrl+C to stop the test server.

### Step 6: Setup Systemd Service

Exit back to root user:

```bash
exit  # Exit deploy user to root
```

Create systemd service:

```bash
nano /etc/systemd/system/tweet-search.service
```

Paste this configuration:

```ini
[Unit]
Description=Tweet Search API
After=network.target

[Service]
Type=simple
User=deploy
WorkingDirectory=/home/deploy/tweet-search
Environment="PATH=/home/deploy/tweet-search/venv/bin"
ExecStart=/home/deploy/tweet-search/venv/bin/python hetzner_backend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit.

Enable and start the service:

```bash
systemctl daemon-reload
systemctl enable tweet-search
systemctl start tweet-search

# Check status
systemctl status tweet-search

# View logs
journalctl -u tweet-search -f
```

Wait for "SERVER READY!" in the logs (database loading takes ~60 seconds).

### Step 7: Setup Nginx Reverse Proxy

Create Nginx configuration:

```bash
nano /etc/nginx/sites-available/tweet-search
```

Paste:

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;  # Or your domain if you have one

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:

```bash
ln -s /etc/nginx/sites-available/tweet-search /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl restart nginx
```

### Step 8: Test API Publicly

From your local machine:

```bash
curl "http://YOUR_SERVER_IP/search?query=bitcoin&limit=5"
```

You should see JSON results! Backend is now live.

## Part 4: Deploy Frontend

### Option A: Deploy to Vercel (Recommended - Easiest & Free)

1. **Push frontend to GitHub** (if not already):

```bash
cd frontend/
git init
git add .
git commit -m "Initial frontend"
gh repo create tweet-search-frontend --public --source=. --push
```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Select your GitHub repo
   - Framework: "Next.js"
   - Set environment variable:
     - Key: `NEXT_PUBLIC_API_URL`
     - Value: `http://YOUR_HETZNER_IP`
   - Click "Deploy"

3. **Done!** Your app will be live at `https://your-app.vercel.app`

### Option B: Deploy Frontend on Same Hetzner Server

If you want everything on one server:

1. **Transfer frontend to Hetzner**:

```bash
# From your local machine
cd frontend/
npm run build
scp -r .next node_modules package.json deploy@YOUR_SERVER_IP:~/tweet-search/frontend/
```

2. **On Hetzner, setup frontend service**:

```bash
# Create systemd service for frontend
sudo nano /etc/systemd/system/tweet-frontend.service
```

Paste:

```ini
[Unit]
Description=Tweet Search Frontend
After=network.target

[Service]
Type=simple
User=deploy
WorkingDirectory=/home/deploy/tweet-search/frontend
Environment="PATH=/home/deploy/tweet-search/frontend/node_modules/.bin:/usr/bin"
Environment="PORT=3000"
Environment="NEXT_PUBLIC_API_URL=http://localhost:8000"
ExecStart=/usr/bin/npm start
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tweet-frontend
sudo systemctl start tweet-frontend
```

3. **Update Nginx to serve both**:

```bash
sudo nano /etc/nginx/sites-available/tweet-search
```

Replace with:

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;

    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API Backend
    location /api {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Restart Nginx:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

4. **Access your app**: `http://YOUR_SERVER_IP`

## Part 5: (Optional) Setup SSL/HTTPS

If you have a domain name:

```bash
# Point your domain A record to YOUR_SERVER_IP first

# Then run certbot
sudo certbot --nginx -d your-domain.com
```

Certbot will automatically configure SSL and update your Nginx config.

## Part 6: Test Everything End-to-End

1. Open your frontend URL (Vercel or Hetzner IP)
2. Type a search query (e.g., "artificial intelligence")
3. Click "Search"
4. You should see results appear!

## Monitoring & Maintenance

### View Backend Logs

```bash
sudo journalctl -u tweet-search -f
```

### View Frontend Logs (if deployed on Hetzner)

```bash
sudo journalctl -u tweet-frontend -f
```

### Restart Services

```bash
# Restart backend
sudo systemctl restart tweet-search

# Restart frontend
sudo systemctl restart tweet-frontend

# Restart Nginx
sudo systemctl restart nginx
```

### Check Resource Usage

```bash
htop  # or top
df -h  # Disk space
free -h  # RAM usage
```

## Cost Comparison

### Before (Modal Always-On)
- Modal min_containers=1: **$267/month**
- Slow reloads every request
- No persistent state

### After (Hetzner)
- Hetzner EX44: **€44/month (~$48/month)**
- Vercel (frontend): **Free**
- **Total: $48/month**
- **Savings: $219/month (82% cheaper!)**
- Fast, persistent, always warm database

## Troubleshooting

### Backend Won't Start
```bash
# Check logs
sudo journalctl -u tweet-search -n 100

# Common issues:
# 1. Missing VOYAGE_API_KEY in .env
# 2. Database files not extracted properly
# 3. Port 8000 already in use
```

### Frontend Can't Connect to Backend
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status
sudo ufw allow 80
sudo ufw allow 443
```

### Database Loading Slow
```bash
# Normal: ~60 seconds to load 40GB into RAM
# Check RAM usage:
free -h

# If RAM is full, you need a server with more RAM
```

## Next Steps

1. Set up monitoring (optional):
   - Install [Netdata](https://www.netdata.cloud/) for real-time monitoring
   - Set up alerting for service downtime

2. Automate backups:
   - Backup `/home/deploy/tweet-search/metadata.pkl` regularly
   - Backup environment variables

3. Scale (if needed):
   - Upgrade to larger Hetzner server (128GB RAM)
   - Add load balancing with multiple backends

## Support

If you encounter issues:
1. Check logs: `sudo journalctl -u tweet-search -f`
2. Verify database files exist: `ls -lh ~/tweet-search/`
3. Test backend directly: `curl http://localhost:8000/health`
4. Check firewall rules: `sudo ufw status`

Congratulations! You now have a production tweet search system running on dedicated hardware for 82% less than Modal!
