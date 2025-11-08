#!/bin/bash
# Hetzner Deployment Script
# Run this on your Hetzner server to deploy the backend

set -e  # Exit on error

echo "================================"
echo "🚀 HETZNER DEPLOYMENT SCRIPT"
echo "================================"
echo ""

# Check if running as deploy user
if [ "$USER" != "deploy" ]; then
    echo "⚠️  Warning: Not running as 'deploy' user. Current user: $USER"
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements_hetzner.txt

echo ""
echo "================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Create .env file with VOYAGE_API_KEY"
echo "2. Make sure database files are in place:"
echo "   - ~/tweet-search/corenn_db/"
echo "   - ~/tweet-search/metadata.pkl"
echo "3. Test the server:"
echo "   cd ~/tweet-search"
echo "   source venv/bin/activate"
echo "   python hetzner_backend.py"
echo ""
echo "4. Setup systemd service (see HETZNER_SETUP.md)"
echo ""
