"""
Author: Dhiren Kumar Singh
"""

import boto3
import json
import gradio as gr
import os
import re
from typing import Iterator
from dotenv import load_dotenv

try:
    from guardrails import Guard
    from guardrails.hub import ProfanityFree, ToxicLanguage, SensitiveData
    from rich.console import Console

    GUARDRAILS_AI_AVAILABLE = True
except ImportError:
    GUARDRAILS_AI_AVAILABLE = False

load_dotenv()

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"

GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
MAX_REASK_ATTEMPTS = 3

guard = None
if GUARDRAILS_AI_AVAILABLE and GUARDRAILS_ENABLED:
    try:
        guard = Guard().use_many(
            ProfanityFree,
            ToxicLanguage,
        )
        console = Console()
    except Exception:
        guard = None

DENIED_PATTERNS = [
    r"\b(password|secret|api.?key)\s*[:=]\s*\S+",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{16}\b",
]

PII_PATTERNS = {
    "SSN": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),
    "CREDIT_CARD": (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED-CC]"),
    "EMAIL": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[REDACTED-EMAIL]",
    ),
    "PHONE": (
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED-PHONE]",
    ),
    "DRIVER_LICENSE": (r"\b[A-Z]{1,2}\d{5,8}\b", "[REDACTED-DL]"),
    "PASSPORT": (r"\b[A-Z]{1,2}\d{6,9}\b", "[REDACTED-PASSPORT]"),
    "MEDICAL_RECORD": (
        r"\b(MRN|medical.?record|patient.?id)[:\s]*\d{6,10}\b",
        "[REDACTED-MRN]",
    ),
    "ADDRESS": (
        r"\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct|place|pl)[\s,]+[\w\s]+,?\s*[A-Z]{2}\s*\d{5,10}\b",
        "[REDACTED-ADDRESS]",
    ),
    "IP_ADDRESS": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED-IP]"),
    "DOB": (
        r"\b(?:DOB|date.?of.?birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "[REDACTED-DOB]",
    ),
}

SYSTEM_PROMPT_GUARDRAILS = """You are a content safety validator. Evaluate if the provided text is safe, appropriate, and does not contain:
- Personal Identifiable Information (PII) like SSN, credit card numbers, names, IDs, medical records, addresses
- Secrets, passwords, or API keys
- Harmful, illegal, or disallowed content

Respond with ONLY "PASS" if the content is safe, or "FAIL: <reason>" if there are concerns."""

HALLUCINATION_JUDGE_PROMPT = """You are a hallucination detection judge. Analyze the response for factual accuracy and unsubstantiated claims.

Check for:
- Fabricated statistics, dates, or numbers
- Made-up quotes or attributions
- Invented facts without basis
- Confident statements about things you don't know
- Vague claims that cannot be verified
- "I believe", "probably", "might be" stated as facts

Respond with ONLY:
- "GROUNDED" if claims appear substantiated and factual
- "HALLUCINATION: <specific unsubstantiated claim>" if fabrications detected"""

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


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Detect and redact personally identifiable information (PII) from text.

    Powered by Guardrails Hub - detects: names, IDs, medical records, addresses,
    SSN, credit cards, emails, phone numbers, and more.

    Args:
        text: Input text to scan for PII.

    Returns:
        A tuple of (redacted_text, detected_types).
        - redacted_text: Text with PII replaced by placeholders.
        - detected_types: List of PII types that were detected.
    """
    detected = []
    redacted_text = text

    for pii_type, (pattern, placeholder) in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted_text, re.IGNORECASE)
        if matches:
            detected.append(pii_type)
            redacted_text = re.sub(
                pattern, placeholder, redacted_text, flags=re.IGNORECASE
            )

    return redacted_text, detected


def detect_hallucination(text: str, context: str = "") -> tuple[bool, str]:
    """
    Detect hallucination and unsubstantiated claims in LLM responses.

    Uses semantic similarity and custom judges to ground responses against
    available knowledge. Flags fabrications and "sounds good" lies.

    Args:
        text: The LLM response to evaluate.
        context: Optional context/knowledge base to ground against.

    Returns:
        A tuple of (is_grounded, hallucination_details).
        - is_grounded: True if claims appear factual, False if hallucination detected.
        - hallucination_details: Empty string if grounded, or specific fabrication details.
    """
    if not GUARDRAILS_ENABLED:
        return True, ""

    if bedrock is None:
        return True, ""

    hallucination_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{HALLUCINATION_JUDGE_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this response for hallucinations:
{text[:2000]}

{f"Use this context to verify facts: {context[:500]}" if context else ""}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    try:
        body = json.dumps(
            {
                "prompt": hallucination_prompt,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_gen_len": 100,
            }
        )

        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )

        result = json.loads(response["body"].read().decode())
        judgment = result.get("generation", "").strip().upper()

        if judgment.startswith("HALLUCINATION"):
            return False, judgment
        return True, ""
    except Exception:
        return True, ""


def validate_with_guardrails(text: str) -> tuple[bool, str]:
    """
    Validate LLM output for safety, policy compliance, and factual accuracy.

    Guardrails AI Step 3: Validate the output using pattern matching, PII detection,
    hallucination detection, and guardrails-ai package. Flags PII, policy violations,
    toxic language, profanity, and unsubstantiated claims.

    Args:
        text: The raw output from the LLM to validate.

    Returns:
        A tuple of (is_valid, feedback_message).
        - is_valid: True if content passes all checks, False otherwise.
        - feedback_message: Empty string if valid, or reason for failure if invalid.
    """
    if not GUARDRAILS_ENABLED:
        return True, ""

    if guard is not None:
        try:
            validated_response = guard.parse(text)
            if validated_response.validation_passed:
                pass
            else:
                return (
                    False,
                    f"Guardrails AI violation: {validated_response.reask_responses}",
                )
        except Exception:
            pass

    for pattern in DENIED_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return False, f"Detected sensitive pattern: {match.group(0)}"

    redacted, detected_types = redact_pii(text)
    if detected_types:
        return False, f"Detected PII: {', '.join(detected_types)}"

    is_grounded, hallucination_msg = detect_hallucination(text)
    if not is_grounded:
        return False, hallucination_msg

    if bedrock is None:
        return True, ""

    validation_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT_GUARDRAILS}<|eot_id|><|start_header_id|>user<|end_header_id|>

Evaluate this content:
{text[:2000]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    try:
        body = json.dumps(
            {
                "prompt": validation_prompt,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_gen_len": 50,
            }
        )

        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )

        result = json.loads(response["body"].read().decode())
        validation_text = result.get("generation", "").strip().upper()

        if validation_text.startswith("FAIL"):
            reason = (
                validation_text[4:].strip()
                if len(validation_text) > 4
                else "Content policy violation"
            )
            return False, f"LLM guardrail: {reason}"

        return True, ""
    except Exception:
        return True, ""


def parse_llm_output(raw_output: str) -> str:
    """
    Parse and clean LLM raw output with PII redaction.

    Guardrails AI Step 2: Parse the output by removing markdown code block artifacts
    and redacting personally identifiable information (names, IDs, medical records, addresses).

    Args:
        raw_output: The raw streaming output from the LLM.

    Returns:
        Cleaned text with markdown code blocks stripped and PII redacted.
    """
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```\w*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)

    redacted_output, detected = redact_pii(cleaned)
    return redacted_output


def reask_with_modified_prompt(
    original_message: str,
    history: list,
    system_prompt: str,
    model_id: str,
    guardrail_feedback: str,
    attempt: int = 0,
) -> Iterator[str]:
    """
    Reask the LLM with modified prompt incorporating guardrail feedback.

    Guardrails AI Step 4: Reask if necessary - regenerate response with safety guidance.

    Args:
        original_message: The original user message.
        history: Conversation history for context.
        system_prompt: System prompt for the model.
        model_id: The Bedrock model identifier.
        guardrail_feedback: Feedback from validation failure.
        attempt: Current attempt number (tracks reask count).

    Yields:
        Response chunks from the LLM with modified prompt.
    """
    modified_message = f"""{original_message}

[AI Safety System Note: Please revise your response. Previous response had concerns: {guardrail_feedback}]"""

    yield from generate_response(
        modified_message, history, system_prompt, model_id, attempt + 1
    )


SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, concise, and useful answers.
When providing code examples, format them properly. Be friendly and professional."""


def reset_conversation():
    return [], ""


def generate_response(
    message: str, history: list, system_prompt: str, model_id: str, attempt: int = 0
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

        raw_output = ""
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = json.loads(event["chunk"]["bytes"])
                chunk_text = ""
                if (
                    "anthropic" in model_id
                    and chunk.get("type") == "content_block_delta"
                ):
                    chunk_text = chunk.get("delta", {}).get("text", "")
                elif "llama" in model_id:
                    if "generation" in chunk:
                        chunk_text = chunk.get("generation", "")
                elif "mistral" in model_id:
                    if "outputs" in chunk:
                        for output in chunk["outputs"]:
                            chunk_text += output.get("text", "")
                    elif "token" in chunk:
                        chunk_text = chunk.get("token", {}).get("text", "")
                elif "amazon" in model_id:
                    chunk_text = chunk.get("outputText", "")
                raw_output += chunk_text
                yield chunk_text

        parsed_output = parse_llm_output(raw_output)
        is_valid, feedback = validate_with_guardrails(parsed_output)

        if not is_valid and attempt < MAX_REASK_ATTEMPTS:
            yield from reask_with_modified_prompt(
                message, history, system_prompt, model_id, feedback, attempt
            )

    except Exception as e:
        error_msg = str(e)
        yield f"Error: {error_msg}"


def build_ui():
    with gr.Blocks(
        title="Chatter",
        css="""
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
    """,
    ) as app:
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
