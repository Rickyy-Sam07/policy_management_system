# 🚀 HackRX API Setup Guide

## ⚡ Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
set GROQ_API_KEY=your_groq_api_key_here
set API_TOKEN=eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d
```

### 3. Start Servers
**Terminal 1:**
```bash
python rtx3050_advanced_api.py
```

**Terminal 2:**
```bash
python make_public.py
```

## 🎯 HackRX Integration

### Authentication Token
- **HackRX Token:** `eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d`
- **Header:** `Authorization: Bearer eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d`

### Request Format
```json
POST /hackrx/run
Content-Type: application/json
Accept: application/json

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["question1", "question2", ...]
}
```

### Response Format
```json
{
    "answers": ["answer1", "answer2", ...]
}
```

## 🔧 Troubleshooting

### 401 Unauthorized
- ✅ **Fixed:** API now accepts HackRX token
- Check environment variable: `echo %API_TOKEN%`

### 405 Method Not Allowed
- Use POST, not GET
- Endpoint: `/hackrx/run`

### Dependencies Missing
```bash
pip install pyngrok psutil python-docx
```

## 📊 Features
- ✅ PDF/DOCX/Email document support
- ✅ 5-stage advanced pipeline
- ✅ RTX 3050 GPU optimization
- ✅ Vector index caching
- ✅ Parallel question processing
- ✅ HackRX token compatibility

## 🎉 Ready for Submission!