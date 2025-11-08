# Hetzner Deployment Guide

Complete guide to deploy the CoreNN tweet search system on Hetzner.

## 1. Order Hetzner Server

Go to [Hetzner Server Auction](https://www.hetzner.com/sb) or order a new dedicated server:

**Recommended Server (for 40GB database):**
- **EX44** (~€44/month)
  - Intel Core i7-6700
  - 64 GB DDR4 RAM
  - 2 x 512 GB NVMe SSD
  - 1 Gbit/s port

**Or check Server Auction for cheaper options with 64GB+ RAM**

## 2. Initial Server Setup

### SSH into your server
```bash
ssh root@YOUR_SERVER_IP
```

### Update system
```bash
apt update && apt upgrade -y
```

### Install Python 3.11+
```bash
apt install -y python3.11 python3.11-venv python3-pip
```

### Install required packages
```bash
apt install -y git curl build-essential
```

### Create deployment user
```bash
adduser deploy
usermod -aG sudo deploy
su - deploy
```

## 3. Setup Application Directory

```bash
mkdir -p ~/tweet-search
cd ~/tweet-search
```

## 4. Transfer Database from Modal

On your LOCAL machine, run:

```bash
# Download database from Modal (40GB)
modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz

# Transfer to Hetzner (replace YOUR_SERVER_IP)
scp corenn_backup.tar.gz deploy@YOUR_SERVER_IP:~/tweet-search/
```

On the HETZNER server:

```bash
cd ~/tweet-search
tar -xzf corenn_backup.tar.gz
# This creates: corenn_db/ and metadata.pkl
```

## 5. Deploy Backend Application

On your LOCAL machine, transfer the backend files:

```bash
scp hetzner_backend.py requirements_hetzner.txt deploy_hetzner.sh deploy@YOUR_SERVER_IP:~/tweet-search/
```

On the HETZNER server:

```bash
cd ~/tweet-search
chmod +x deploy_hetzner.sh
./deploy_hetzner.sh
```

## 6. Setup Environment Variables

Create `.env` file:

```bash
nano ~/tweet-search/.env
```

Add:
```
VOYAGE_API_KEY=your_voyage_api_key_here
PORT=8000
```

## 7. Setup Systemd Service (Run Backend as Service)

Create service file:

```bash
sudo nano /etc/systemd/system/tweet-search.service
```

Add:
```ini
[Unit]
Description=Tweet Search API
After=network.target

[Service]
Type=simple
User=deploy
WorkingDirectory=/home/deploy/tweet-search
Environment="PATH=/home/deploy/tweet-search/venv/bin"
ExecStart=/home/deploy/tweet-search/venv/bin/uvicorn hetzner_backend:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tweet-search
sudo systemctl start tweet-search
sudo systemctl status tweet-search
```

## 8. Setup Nginx Reverse Proxy

Install Nginx:
```bash
sudo apt install -y nginx
```

Create Nginx config:
```bash
sudo nano /etc/nginx/sites-available/tweet-search
```

Add:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

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

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tweet-search /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 9. (Optional) Setup SSL with Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 10. Test API

```bash
curl "http://YOUR_SERVER_IP/search?query=bitcoin&limit=5"
```

## 11. Deploy Frontend

The frontend can be deployed on:
- **Vercel** (recommended, free)
- **Netlify** (free)
- **Same Hetzner server** (serve with Nginx)

Point the frontend API calls to: `http://YOUR_SERVER_IP/search`

## Monitoring & Maintenance

### View logs
```bash
sudo journalctl -u tweet-search -f
```

### Restart service
```bash
sudo systemctl restart tweet-search
```

### Check resource usage
```bash
htop
```

## Cost Summary

- **Hetzner EX44**: €44/month (~$48/month)
- **Domain** (optional): ~$10/year
- **Total**: ~$48/month

Compare to Modal's always-on approach: $267/month

**Savings: $219/month! 🎉**
