"""
Setup and Installation Script
============================

Automated setup for the Document Processing System.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "temp_attachments",
        "test_files",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories setup complete")

def check_system_dependencies():
    """Check for system-level dependencies."""
    print("\n🔍 Checking system dependencies...")
    
    # Check for Tesseract (optional)
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract found: {version}")
        else:
            print("⚠️  Tesseract not found (OCR will use PaddleOCR only)")
    except FileNotFoundError:
        print("⚠️  Tesseract not found (OCR will use PaddleOCR only)")
        
        if platform.system() == "Windows":
            print("   💡 To install Tesseract on Windows:")
            print("      1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("      2. Install and add to PATH")
            print("      3. Restart command prompt")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        from document_detector import DocumentTypeDetector
        print("✅ Document detector import successful")
        
        # Test detector initialization
        detector = DocumentTypeDetector()
        print("✅ Document detector initialization successful")
        
        # Test basic functionality
        test_file = Path("test_files/sample.txt")
        if test_file.exists():
            result = detector.detect_document_type(test_file)
            print(f"✅ Document detection test successful: {result.metadata.source_type}")
        else:
            print("⚠️  No test file available for detection test")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    print("\n⚙️  Creating sample configuration...")
    
    config_content = """
# Document Processing System Configuration
# ======================================

[general]
# Default OCR language (en, chi_sim, fra, deu, etc.)
default_ocr_language = en

# Maximum file size for processing (in MB)
max_file_size_mb = 100

# Enable preprocessing for OCR
enable_image_preprocessing = true

# Log level (DEBUG, INFO, WARNING, ERROR)
log_level = INFO

[pdf]
# Preferred PDF parser (pdfplumber, pymupdf)
preferred_parser = pdfplumber

# OCR confidence threshold for fallback
ocr_fallback_threshold = 0.6

[ocr]
# Preferred OCR engine (paddleocr, tesseract)
preferred_engine = paddleocr

# Enable multiple OCR engines for verification
enable_multi_engine = false

# OCR confidence threshold
confidence_threshold = 0.5

[api]
# API server host
host = 0.0.0.0

# API server port
port = 8000

# Enable CORS
enable_cors = true

# Maximum batch size for processing
max_batch_size = 50

# Request timeout (seconds)
request_timeout = 300

[directories]
# Temporary files directory
temp_dir = temp_files

# Logs directory
logs_dir = logs

# Exports directory
exports_dir = exports

# Attachments extraction directory
attachments_dir = temp_attachments
"""
    
    config_file = Path("config.ini")
    with open(config_file, 'w') as f:
        f.write(config_content.strip())
    
    print(f"✅ Configuration file created: {config_file}")

def main():
    """Main setup function."""
    print("🚀 Document Processing System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup directories
    setup_directories()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Create sample configuration
    create_sample_config()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed")
        print("   Please check the error messages above and resolve any issues.")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Review config.ini and adjust settings as needed")
    print("   2. Test with: python example_usage.py")
    print("   3. Start API with: python api.py")
    print("   4. Access docs at: http://localhost:8000/docs")
    print("\n📚 Documentation available in README.md")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
