#!/bin/bash

echo "======================================================================"
echo "  Taka-Hero AI Priority Classification System - Quick Setup"
echo "======================================================================"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data/waste_images
mkdir -p data/training
echo "✅ Directories created"
echo ""

# Check Python version
echo "🐍 Checking Python..."
python3 --version
echo ""

# Install basic dependencies
echo "📦 Installing basic dependencies..."
pip install pillow
echo "✅ Basic dependencies installed"
echo ""

# Check if PyTorch should be installed
echo "Would you like to install PyTorch for deep learning? (y/n)"
echo "Note: The system works fine without it using rule-based classification"
read -r install_pytorch

if [ "$install_pytorch" = "y" ] || [ "$install_pytorch" = "Y" ]; then
    echo "⚙️  Installing PyTorch (this may take a while)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers
    echo "✅ PyTorch installed"
else
    echo "⏭️  Skipping PyTorch installation (using rule-based classification)"
fi
echo ""

# Test the predictor
echo "🧪 Testing the AI predictor..."
python3 waste_predictor.py
echo ""

# Check if database needs migration
echo "🗄️  Database setup..."
if [ -f "instance/taka_hero.db" ]; then
    echo "⚠️  Existing database found!"
    echo "The database needs new columns for AI features."
    echo "Would you like to:"
    echo "1. Reset database (WARNING: Deletes all data)"
    echo "2. Skip (add columns manually later)"
    read -r db_choice
    
    if [ "$db_choice" = "1" ]; then
        rm instance/taka_hero.db
        echo "✅ Database reset"
    else
        echo "⏭️  Keeping existing database (remember to add AI columns)"
    fi
else
    echo "✅ No existing database found - will be created on first run"
fi
echo ""

# Final instructions
echo "======================================================================"
echo "  🎉 Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the app:"
echo "   python3 app.py"
echo ""
echo "2. Submit a test report to see AI classification in action"
echo ""
echo "3. (Optional) Train the deep learning model:"
echo "   jupyter notebook waste_priority_model_training.ipynb"
echo ""
echo "4. Read the full guide:"
echo "   cat AI_MODEL_GUIDE.md"
echo ""
echo "======================================================================"
echo "Making Kenya clean! 🇰🇪♻️"
echo "======================================================================"
