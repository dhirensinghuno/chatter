<!-- Author: Dhiren Kumar Singh -->

# Chatter - AI Chat Application

A Gradio-based chat application powered by Amazon Bedrock (Llama, Mistral models).

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials
Create a `.env` file in the project directory:
```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
```

**To get AWS credentials:**
1. Sign in to AWS Console: https://console.aws.amazon.com
2. Go to IAM → Users → Create User
3. Attach policy: `AmazonBedrockFullAccess` or create custom policy with `bedrock:InvokeModel`
4. Create Access Key from Security Credentials

### 3. Enable Bedrock model access
1. Go to AWS Console → Amazon Bedrock → Model access
2. Click "Manage model access"
3. Enable models:
   - **Meta** → Llama 3.1 70B Instruct
   - **Mistral AI** → Mistral Large
4. Click "Request access"

## Usage

```bash
python chat.py
```

Open http://localhost:7861 in your browser.

## Features

- Streaming responses
- Conversation history (persists during session)
- Multiple model support (Llama 3 70B, Llama 3 8B, Mistral Large)
- Customizable system prompt
- Clean, modern UI

## Troubleshooting

**Error: "on-demand throughput isn't supported"**
- Use inference profile IDs (e.g., `us.meta.llama3-1-70b-instruct-v1:0`)

**Error: "AccessDeniedException"**
- Verify IAM user has `bedrock:InvokeModel` permission
- Check model is enabled in Bedrock Model access

**Error: "Malformed input request"**
- Ensure you're using correct request format for each model
