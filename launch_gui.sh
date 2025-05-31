#!/bin/bash

# AI Blob Interactive GUI Launcher
# This script launches the Streamlit web interface for interactive clip selection

echo "🎬 AI Blob - Interactive Video Generator"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the AI-Blob-MM16 root directory"
    exit 1
fi

# Check if virtual environment is activated (optional)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "💡 Tip: Consider activating your virtual environment first"
    echo "   Example: source venv/bin/activate"
    echo ""
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Find available port
PORT=8501
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    PORT=$((PORT+1))
done

echo "🚀 Starting GUI on port $PORT..."
echo "📱 Access the interface at: http://localhost:$PORT"
echo ""
echo "🔧 Usage:"
echo "   1. Enter your video theme"
echo "   2. Select clips interactively or use 'I'm feeling lucky'"
echo "   3. Export when complete"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Set Python path to include current directory for module imports
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Launch Streamlit
streamlit run src/gui/streamlit_app.py --server.port $PORT --server.headless true
