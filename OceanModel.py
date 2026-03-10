import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your saved/merged model folder
# Change this to wherever you saved the model (local path or mounted Drive path)
MODEL_PATH = "./ocean_model_merged"   # e.g. "/content/drive/MyDrive/ocean_model_merged"

SYSTEM_PROMPT = (
    "You are an objective psychometrician and career counselor. "
    "You analyze Big Five (OCEAN) personality scores and produce structured reports. "
    "Scores are on a 0-100 normalized scale where 50 is average. "
    "Be honest, direct, and use plain language. Do not sugarcoat weaknesses. "
    "You MUST always respond using EXACTLY this markdown structure, no deviations:\n\n"
    "## Personality Overview\n"
    "2-3 sentences summarizing the core personality type.\n\n"
    "## Key Strengths\n"
    "- Strength 1\n"
    "- Strength 2\n"
    "- Strength 3\n\n"
    "## Personality Traits\n"
    "- **Openness:** one sentence interpretation\n"
    "- **Conscientiousness:** one sentence interpretation\n"
    "- **Extraversion:** one sentence interpretation\n"
    "- **Agreeableness:** one sentence interpretation\n"
    "- **Neuroticism:** one sentence interpretation\n\n"
    "## Blind Spots\n"
    "- Blind spot 1\n"
    "- Blind spot 2\n\n"
    "## Work & Career Style\n"
    "2-3 sentences on how this person operates professionally.\n\n"
    "## Key Recommendations\n"
    "- Recommendation 1\n"
    "- Recommendation 2\n"
    "- Recommendation 3"
)

# Model is lazy-loaded once and cached — Streamlit reruns won't reload it
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            _model = _model.to(device)
        _model.eval()
    return _tokenizer, _model


def llm_analysis(scores):
    """
    Generates a personality report using the locally fine-tuned model.

    Args:
        scores (dict): Dictionary of OCEAN scores (e.g., {'Openness': 72.5, ...})

    Returns:
        str: Markdown formatted report.
    """
    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    scores_text = "\n".join([f"- {trait}: {score}" for trait, score in scores.items()])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            "Generate a personality report for the following OCEAN scores.\n\n"
            f"{scores_text}\n\n"
            "Use EXACTLY the markdown schema defined: Personality Overview, Key Strengths, "
            "Personality Traits, Blind Spots, Work & Career Style, Key Recommendations."
        )},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
