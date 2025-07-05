import os
import subprocess
import threading
import time
from pathlib import Path

# Set environment variables BEFORE importing your main app
os.environ.setdefault('CONFLUENCE_USERNAME', os.getenv('CONFLUENCE_USERNAME', ''))
os.environ.setdefault('CONFLUENCE_API_TOKEN', os.getenv('CONFLUENCE_API_TOKEN', ''))
os.environ.setdefault('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
os.environ.setdefault('ADMIN_USERNAME', os.getenv('ADMIN_USERNAME', 'admin'))
os.environ.setdefault('ADMIN_PASSWORD', os.getenv('ADMIN_PASSWORD', 'admin123'))
os.environ.setdefault('STUDENT_USERNAME', os.getenv('STUDENT_USERNAME', 'student'))
os.environ.setdefault('STUDENT_PASSWORD', os.getenv('STUDENT_PASSWORD', 'student123'))

# Modify port for HF Spaces
os.environ.setdefault('PORT', '7860')  # HF Spaces uses port 7860

def run_fastapi():
    """Run the FastAPI application"""
    try:
        # Import and run your main application
        import uvicorn
        from main import app  # Import your FastAPI app
        
        # Run on HF Spaces port
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=7860,  # HF Spaces required port
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting FastAPI: {e}")

if __name__ == "__main__":
    print("🚀 Starting APBot on Hugging Face Spaces...")
    print("📡 Server will be available at: https://yourusername-apbot.hf.space")
    
    # Start FastAPI server
    run_fastapi()