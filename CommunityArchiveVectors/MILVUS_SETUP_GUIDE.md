# Complete Guide: Self-Hosting Milvus for Tweet Embeddings

## Part 1: Install Docker (if not already installed)

### For macOS:
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Verify installation:
   ```bash
   docker --version
   docker compose version
   ```

### For Linux:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

---

## Part 2: Install Milvus with Docker Compose

### Step 1: Download Milvus Docker Compose file
```bash
cd /Users/noasouccar/Desktop/FractalProjects/CommunityArchiveVectors/CommunityArchiveVectors

curl -L https://github.com/milvus-io/milvus/releases/download/v2.6.4/milvus-standalone-docker-compose.yml -o docker-compose.yml
```

✅ **This file is already downloaded for you!**

### Step 2: Start Milvus
```bash
docker compose up -d
```

This will start 3 containers:
- `milvus-etcd` - Metadata storage
- `milvus-minio` - Object storage
- `milvus-standalone` - Main Milvus server

### Step 3: Verify Milvus is running
```bash
docker compose ps
```

You should see all 3 containers with status "Up"

### Step 4: Check logs (if needed)
```bash
docker compose logs milvus-standalone
```

### Step 5: Access Milvus Web UI (optional)
Open in browser: http://localhost:9091/webui/

---

## Part 3: Install Milvus SDK for Node.js

```bash
bun add @zilliz/milvus2-sdk-node
```

---

## Part 4: Create Collection and Upload Your Tweet Embeddings

I'll create a script for you in the next step that will:
1. Connect to your local Milvus instance
2. Create a collection for tweet embeddings
3. Load your Voyage AI embeddings (1024 dimensions)
4. Enable vector search

---

## Milvus Connection Details

Once running, your Milvus instance will be accessible at:
- **Address**: `localhost:19530`
- **Web UI**: http://localhost:9091/webui/
- **MinIO Console** (object storage): http://localhost:9001/

Default credentials for MinIO:
- Username: `minioadmin`
- Password: `minioadmin`

---

## Useful Docker Commands

### Stop Milvus
```bash
docker compose down
```

### Stop and remove all data
```bash
docker compose down
rm -rf volumes/
```

### Restart Milvus
```bash
docker compose restart
```

### View resource usage
```bash
docker stats
```

---

## Troubleshooting

### Container won't start?
```bash
# Check logs
docker compose logs milvus-standalone

# Check all container logs
docker compose logs
```

### Port already in use?
Edit `docker-compose.yml` and change port mappings:
```yaml
ports:
  - "19530:19530"  # Change first number to something else like "19531:19530"
```

### Out of memory?
In Docker Desktop:
- Go to Settings → Resources
- Increase Memory to at least 4GB (8GB recommended)

---

## Next Steps

After you have Milvus running (Steps 1-2), let me know and I'll:
1. Create a Node.js script to upload your tweet embeddings
2. Show you how to perform vector searches
3. Compare search performance vs Supabase

Ready to start? Run these commands:

```bash
# 1. Make sure Docker is installed
docker --version

# 2. Start Milvus
docker compose up -d

# 3. Verify it's running
docker compose ps
```
