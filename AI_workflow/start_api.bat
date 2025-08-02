@echo off
echo 🚀 Starting HackRX API Server...
echo.

REM Set environment variables
set GROQ_API_KEY=your_groq_api_key_here
set API_TOKEN=eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d

echo ✅ Environment variables set
echo 📡 GROQ_API_KEY: %GROQ_API_KEY:~0,10%...
echo 🔑 API_TOKEN: %API_TOKEN:~0,10%...
echo.

echo 🏃 Starting FastAPI server on port 8001...
python rtx3050_advanced_api.py

pause