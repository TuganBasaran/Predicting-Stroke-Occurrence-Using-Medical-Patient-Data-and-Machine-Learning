#!/bin/bash

# Stroke Prediction Project - Quick Start Script

echo "🏥 Stroke Prediction Project Setup"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: jupyter notebook stroke_prediction_full.ipynb"
echo "      (or open it in VS Code)"
echo "   2. Execute all cells to train models"
echo "   3. Run: streamlit run app.py"
echo ""
echo "📖 For detailed instructions, see SETUP_GUIDE.md"
