#!/bin/bash

# Football Player Salary Prediction - Quick Setup Script
# This script sets up the entire project in one command

set -e

echo "FOOTBALL SALARY PREDICTION - QUICK SETUP"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo ""
echo "[1/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Create project structure
echo ""
echo "[2/6] Creating project structure..."
mkdir -p data
mkdir -p models
mkdir -p notebooks
mkdir -p src
mkdir -p deployment/cloud
mkdir -p deployment/kubernetes

echo -e "${GREEN}✓${NC} Project structure created"

# Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[4/6] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓${NC} Dependencies installed"

# Verify installation
echo ""
echo "[6/6] Verifying installation..."

# Check if key packages are installed
python3 -c "import pandas; import sklearn; import flask; import xgboost" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All required packages installed successfully"
else
    echo "Warning: Some packages may not be installed correctly"
fi

# Summary
echo ""
echo "SETUP COMPLETE!"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "2. Place your wages.csv file in the data/ directory:"
echo "   ${BLUE}cp /path/to/wages.csv data/${NC}"
echo ""
echo "3. Train the model:"
echo "   ${BLUE}python src/train.py${NC}"
echo ""
echo "4. Run the API server:"
echo "   ${BLUE}python src/predict.py${NC}"
echo ""
echo "5. Test the API:"
echo "   ${BLUE}python test_api.py${NC}"
echo ""
echo "Or use the Makefile:"
echo "   ${BLUE}make train${NC}  - Train the model"
echo "   ${BLUE}make run${NC}    - Run the API"
echo "   ${BLUE}make test${NC}   - Test the API"
echo ""
echo "For Docker deployment:"
echo "   ${BLUE}make docker-build${NC}"
echo "   ${BLUE}make docker-run${NC}"
echo ""
