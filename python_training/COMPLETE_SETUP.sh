#!/bin/bash
# Complete Setup Script - Installs everything from scratch

echo "============================================================"
echo "  COMPLETE SETUP - Installing Everything"
echo "============================================================"

# Check Python
echo ""
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Install Python 3.9+ first"
    exit 1
fi
python3 --version
echo "✅ Python found"

# Check pip
echo ""
echo "[2/5] Checking pip..."
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --default-pip
fi
pip --version || pip3 --version
echo "✅ pip found"

# Install PyTorch with CUDA
echo ""
echo "[3/5] Installing PyTorch with CUDA support..."
echo "   This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✅ PyTorch installed"

# Install other dependencies
echo ""
echo "[4/5] Installing other dependencies..."
pip install numpy pandas boto3 python-dotenv supabase gymnasium tqdm

echo "✅ Dependencies installed"

# Verify GPU
echo ""
echo "[5/5] Verifying GPU access..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Now run: python train.py"
