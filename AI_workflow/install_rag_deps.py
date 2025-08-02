#!/usr/bin/env python3
"""
Install additional dependencies for RAG pipeline
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install RAG dependencies"""
    print("🚀 Installing RAG pipeline dependencies...")
    
    # Additional packages needed for RAG
    packages = [
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "faiss-cpu>=1.7.4"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation complete: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("✅ All RAG dependencies installed successfully!")
        print("🚀 Ready to run GPU-accelerated RAG pipeline")
    else:
        print("⚠️ Some packages failed to install. Check the errors above.")

if __name__ == "__main__":
    main()