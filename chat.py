import boto3
import json
import gradio as gr
import os
from typing import Iterator
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

bedrock = None
if aws_access_key_id and aws_secret_access_key:
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, concise, and useful answers.
When providing code examples, format them properly. Be friendly and professional."""


def reset_conversation():
    return [], ""


def generate_response(
    message: str, history: list, system_prompt: str, model_id: str
) -> Iterator[str]:
    global bedrock
    if bedrock is None:
        yield "⚠️ Please configure AWS credentials first"
        return

    messages = []
    if history:
        for item in history:
            if isinstance(item, dict):
                messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                messages.append({"role": "user", "content": item[0]})
                messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": message})

    if "anthropic" in model_id:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": messages,
            }
        )
    elif "llama" in model_id:
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            role_name = "user" if role == "user" else "assistant"
            prompt += f"<|start_header_id|>{role_name}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        body = json.dumps(
            {
                "prompt": prompt,
                "temperature": 0.5,
                "top_p": 0.9,
                "max_gen_len": 1024,
            }
        )
    elif "mistral" in model_id:
        prompt = f"<s>[INST] {system_prompt} [/INST]"
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            prompt += (
                f" [INST] {m['content']} [/INST]"
                if role == "user"
                else f" {m['content']}"
            )
        body = json.dumps({"prompt": prompt, "max_tokens": 1024, "temperature": 0.7})
    else:
        body = json.dumps(
            {
                "inputText": f"{system_prompt}\n\nUser: {message}",
                "textGenerationConfig": {"maxTokenCount": 1024},
            }
        )

    try:
        response = bedrock.invoke_model_with_response_stream(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )

        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = json.loads(event["chunk"]["bytes"])
                if (
                    "anthropic" in model_id
                    and chunk.get("type") == "content_block_delta"
                ):
                    yield chunk.get("delta", {}).get("text", "")
                elif "llama" in model_id:
                    if "generation" in chunk:
                        yield chunk.get("generation", "")
                elif "mistral" in model_id:
                    if "outputs" in chunk:
                        for output in chunk["outputs"]:
                            yield output.get("text", "")
                    elif "token" in chunk:
                        yield chunk.get("token", {}).get("text", "")
                elif "amazon" in model_id:
                    yield chunk.get("outputText", "")
    except Exception as e:
        error_msg = str(e)
        yield f"Error: {error_msg}"


def build_ui():
    with gr.Blocks(title="Chatter",         css="""
        #message-input textarea {
            font-size: 16px !important;
            border-radius: 12px !important;
            border: 2px solid #e0e0e0 !important;
            padding: 12px !important;
        }
        #message-input textarea:focus {
            border-color: #007bff !important;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.2) !important;
        }
    """) as app:
        gr.Markdown("# Chatter")

        with gr.Accordion("Settings", open=False, visible=False):
            system_prompt = gr.Textbox(
                label="System Prompt", value=SYSTEM_PROMPT, lines=3
            )
            model_selector = gr.Dropdown(
                label="Model",
                choices=[
                    ("Llama 3 70B", "us.meta.llama3-1-70b-instruct-v1:0"),
                    ("Llama 3 8B", "us.meta.llama3-1-8b-instruct-v1:0"),
                    ("Mistral Large", "us.mistral.mistral-large-2402-v1:0"),
                ],
                value=MODEL_ID,
            )

        chatbot = gr.Chatbot(height=450)
        msg = gr.Textbox(
            placeholder="Type your message here...",
            lines=5,
            show_label=False,
            container=True,
            text_align="left",
            scale=1,
            elem_id="message-input",
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary", scale=3, size="lg")
            clear_btn = gr.Button("Clear", scale=1, size="lg")

        def respond(message, history, system, model_id):
            if not message.strip():
                return history, ""

            if history is None:
                history = []

            full_response = ""
            for chunk in generate_response(message, history, system, model_id):
                full_response += chunk
                new_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_response},
                ]
                yield new_history, ""

        submit_btn.click(
            respond,
            inputs=[msg, chatbot, system_prompt, model_selector],
            outputs=[chatbot, msg],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, system_prompt, model_selector],
            outputs=[chatbot, msg],
        )
        clear_btn.click(reset_conversation, outputs=[chatbot, msg])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0", server_port=7861, theme=gr.themes.Soft(), max_threads=5
    )
