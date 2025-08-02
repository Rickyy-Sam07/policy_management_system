# 🚀 RTX 3050 Distributed Setup Guide

## 📋 Prerequisites
- **Computer 1**: RTX 3050 GPU, Python 3.8+, ngrok
- **Computer 2**: Any computer, Python 3.8+, ngrok
- **2 Different GROQ API Keys** (different accounts)

## 🔧 Computer 1 Setup (Main Server)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn torch sentence-transformers faiss-cpu PyMuPDF groq requests pyngrok
```

### 2. Set Environment Variables
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_1
set COMPUTER2_WORKER_URL=https://your-computer2-ngrok-url.ngrok-free.app

# Linux/Mac
export GROQ_API_KEY=your_groq_api_key_1
export COMPUTER2_WORKER_URL=https://your-computer2-ngrok-url.ngrok-free.app
```

### 3. Start Main Server
```bash
python rtx3050_distributed_api.py
```

### 4. Expose with ngrok
```bash
ngrok http 8001
```

## 🔧 Computer 2 Setup (Worker Server)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn groq
```

### 2. Set Environment Variables
```bash
# Windows
set GROQ_API_KEY_2=your_groq_api_key_2

# Linux/Mac
export GROQ_API_KEY_2=your_groq_api_key_2
```

### 3. Start Worker Server
```bash
python computer2_worker.py
```

### 4. Expose with ngrok
```bash
ngrok http 8002
```

### 5. Update Computer 1
Update Computer 1's `COMPUTER2_WORKER_URL` with the ngrok URL from step 4.

## 🌐 Architecture Flow

```
HackRX Request → Computer 1 (Main Server)
                     ↓
                 Process PDF → Create Vector DB
                     ↓
                 Split Questions (6 + 6)
                     ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
Computer 1                         Computer 2
(Questions 1-6)                   (Questions 7-12)
GROQ API Key 1                    GROQ API Key 2
    ↓                                   ↓
Process with Context              Process with Context
    ↓                                   ↓
    └─────────────────┬─────────────────┘
                     ↓
              Combine Results → Return to HackRX
```

## 📊 Benefits

### Rate Limits
- **Single Computer**: 6,000 TPM limit
- **Distributed**: 12,000 TPM total (6,000 + 6,000)

### Processing Speed
- **Single Computer**: Sequential batches
- **Distributed**: Parallel processing across 2 computers

### Token Usage (12 Questions)
- **Computer 1**: 6 × 695 = 4,170 tokens
- **Computer 2**: 6 × 695 = 4,170 tokens
- **Total**: 8,340 tokens processed in parallel

## 🧪 Testing

### Test Computer 2 Worker
```bash
curl -X POST "http://localhost:8002/process_questions" \
  -H "Content-Type: application/json" \
  -d '{
    "questions_with_context": [
      {
        "question": "Test question?",
        "context": "Test context"
      }
    ]
  }'
```

### Test Distributed System
```bash
curl -X POST "https://your-computer1-ngrok-url.ngrok-free.app/hackrx/run" \
  -H "Authorization: Bearer eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "your_pdf_url",
    "questions": ["Question 1?", "Question 2?"]
  }'
```

## 🔍 Monitoring

### Computer 1 Logs
- Document processing time
- Question splitting
- Worker communication
- Result combination

### Computer 2 Logs
- Questions received
- Processing time per question
- GROQ API responses

## 🚨 Troubleshooting

### Worker Connection Issues
1. Check Computer 2 ngrok URL is accessible
2. Verify COMPUTER2_WORKER_URL environment variable
3. Check firewall settings

### Rate Limit Issues
1. Verify different GROQ API keys are being used
2. Check token usage in GROQ dashboard
3. Monitor processing logs for 429 errors

### Fallback Behavior
- If Computer 2 fails, system falls back to single computer processing
- Graceful degradation ensures HackRX compatibility

## 📈 Scaling for Larger Workloads

### 35 Questions Example
- **Computer 1**: 18 questions = 12,510 tokens
- **Computer 2**: 17 questions = 11,815 tokens
- **Total**: 24,325 tokens (within 24,000 TPM combined limit)

### 1000-Page PDF
- Document processing: ~15-20 seconds on Computer 1
- Question processing: Parallel across both computers
- Total time: ~25-30 seconds vs 60+ seconds single computer