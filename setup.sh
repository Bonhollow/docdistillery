#!/bin/bash

# DocDistillery Unified Setup Script
# Handles system dependencies (macOS) and Python environment.

set -e

echo "🚀 Starting DocDistillery Setup..."

# 1. System Dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS. Checking for Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "⚠️ Homebrew not found. Please install it from https://brew.sh/ and run this script again."
    else
        echo "📦 Installing system dependencies (pango, cairo, libffi) for PDF export..."
        brew install pango || echo "⚠️ Failed to install pango. PDF export might not work."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux. Checking for package manager..."
    if command -v apt-get &> /dev/null; then
        echo "📦 Using apt-get to install dependencies..."
        sudo apt-get update && sudo apt-get install -y libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libffi-dev libjpeg-dev libopenjp2-7-dev
    elif command -v dnf &> /dev/null; then
        echo "📦 Using dnf to install dependencies..."
        sudo dnf install -y pango cairo gdk-pixbuf2 libffi-devel
    elif command -v pacman &> /dev/null; then
        echo "📦 Using pacman to install dependencies..."
        sudo pacman -Sy --noconfirm pango cairo libffi
    else
        echo "⚠️ Unsupported package manager. Please manually install pango, cairo, and libffi."
    fi
else
    echo "ℹ️ Unsupported OS ($OSTYPE). Please ensure pango, cairo, and libffi are installed manually."
fi

# 2. Python Virtual Environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Created .venv"
else
    echo "ℹ️ .venv already exists."
fi

# 3. Installing Python Dependencies
echo "🛠️ Installing Python packages..."
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "✨ Setup Complete!"
echo "-------------------------------------------------------"
echo "To start using DocDistillery, activate your environment:"
echo "source .venv/bin/activate"
echo ""
echo "Try running: docdistillery --help"
echo "-------------------------------------------------------"
