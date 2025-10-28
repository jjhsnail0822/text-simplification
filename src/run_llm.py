import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from pathlib import Path
import sys

def main():
    # Load environment variables from .env.local file
    env_path = Path('.env.local')
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="Run LLM to simplify text to a certain vocabulary level.")
    parser.add_argument('--model', type=str, default='google/gemma-3-12b-it', help='Model ID to use (HuggingFace or gemini-2.5-flash)')
    parser.add_argument('--language', type=str, choices=['en', 'ja', 'ko', 'zh'], default='en', help='Language of the text')
    parser.add_argument('--gpu', type=int, help='Number of GPUs to use (only for vLLM/HF models)')
    parser.add_argument('--max_tokens', type=int, default=16384, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (used for Gemini; HF path fixed at 0 unless changed manually)')
    args = parser.parse_args()

    torch.manual_seed(42)

    model_id = args.model
    model_id_name = model_id.split('/')[-1]
    is_gemini = model_id.startswith("gemini")

    data_dir = f"data/wikipedia/parsed_wikitext/{args.language}"
    files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.json')]
    ds = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ds.append(data)

    levels = {
        'en': ['CEFR A1', 'CEFR A2', 'CEFR B1', 'CEFR B2', 'CEFR C1', 'CEFR C2'],
        'ja': ['JLPT N5', 'JLPT N4', 'JLPT N3', 'JLPT N2', 'JLPT N1'],
        'ko': ['TOPIK I', 'TOPIK II'],
        'zh': ['HSK 3.0 Level 1', 'HSK 3.0 Level 2', 'HSK 3.0 Level 3', 'HSK 3.0 Level 4', 'HSK 3.0 Level 5', 'HSK 3.0 Level 6', 'HSK 3.0 Level 7-9']
    }

    results = []
    output_path = f"results/{args.language}/{model_id_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results = loaded
        except (json.JSONDecodeError, OSError):
            pass

    # Build fast lookup dictionary for resume logic (title -> entry)
    title_to_entry = {}
    for entry in results:
        if isinstance(entry, dict) and "title" in entry:
            title_to_entry[entry["title"]] = entry

    if is_gemini:
        from google import genai
        from google.genai import types

        # Support multiple API keys rotation (comma-separated) similar to reference
        api_keys_env = os.getenv("GOOGLE_API_KEYS") or os.getenv("GOOGLE_API_KEY")
        if not api_keys_env:
            raise ValueError("Gemini selected but no GOOGLE_API_KEYS or GOOGLE_API_KEY env var found.")
        api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
        if not api_keys:
            raise ValueError("No valid Gemini API keys parsed.")
        current_key_index = 0

        def init_gemini_client():
            """Initialize Gemini client with current API key."""
            key = api_keys[current_key_index]
            print(f"[Gemini] Using API key index {current_key_index}")
            return genai.Client(api_key=key)

        client = init_gemini_client()

        def generate_with_gemini(prompt: str):
            """Generate text using Gemini with key rotation on failure.
            Tries each key once. If all keys fail, raises RuntimeError.
            """
            nonlocal client, current_key_index
            last_error = None
            for _ in range(len(api_keys)):
                try:
                    generation_config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        maxOutputTokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[prompt],
                        config=generation_config,
                    )
                    text = (response.text or "").strip()
                    if not text:
                        diag_parts = []
                        try:
                            candidates = getattr(response, "candidates", None)
                            if candidates and len(candidates) > 0:
                                c0 = candidates[0]
                                fr = getattr(c0, "finish_reason", None) or getattr(c0, "finishReason", None)
                                if fr:
                                    diag_parts.append(f"finish_reason={fr}")
                                safety = getattr(c0, "safety_ratings", None) or getattr(c0, "safetyRatings", None)
                                if safety:
                                    diag_parts.append(f"safety_ratings={safety}")
                            pf = getattr(response, "prompt_feedback", None) or getattr(response, "promptFeedback", None)
                            if pf:
                                diag_parts.append(f"prompt_feedback={pf}")
                        except Exception:
                            pass
                        diag_msg = "; ".join(diag_parts) if diag_parts else f"raw_response={repr(response)[:500]}"
                        raise RuntimeError(f"Empty response text. Diagnostics: {diag_msg}")
                    return text
                except Exception as e:
                    last_error = e
                    print(f"[Gemini Error @ key {current_key_index}] {e}")
                    # rotate to next key and retry
                    if current_key_index < len(api_keys) - 1:
                        current_key_index += 1
                    else:
                        current_key_index = 0
                    client = init_gemini_client()
                    continue
            raise RuntimeError(f"All API keys failed. Last error: {last_error}")

    else:
        # Initialize vLLM / HF model path
        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        llm = LLM(model_id, max_model_len=args.max_tokens, tensor_parallel_size=args.gpu if args.gpu else 1)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save throttling: save after every 10 updated entries
    updated_entries_since_save = 0

    for wikitext in tqdm(ds):
        title = wikitext['title']

        # If entry already exists, reuse it; else create new
        if title in title_to_entry:
            entry = title_to_entry[title]
        else:
            entry = {
                "title": title,
                "original": wikitext['shortened_text'],
                "simplified": {}
            }
            results.append(entry)
            title_to_entry[title] = entry

        # Set of levels already generated (successful or not)
        existing_levels = set(entry.get("simplified", {}).keys())

        # If all levels are already present, skip (resume behavior)
        required_levels = set(levels[args.language])
        if required_levels.issubset(existing_levels):
            continue  # Already fully processed

        updated = False  # Track whether we added new generations this pass

        for level in levels[args.language]:
            # Skip level if it already exists (you can refine: retry errors only)
            if level in existing_levels:
                continue

            user_instruction = (
                f"You are a careful rewrite assistant.\n"
                f"Rewrite the <TEXT> so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\n"
                f"Replace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\n"
                f"Do not skip, shorten, or omit any part of the text. Keep sentence count and structure.\n"
                f"Output only the fully converted text with no explanations, instructions, or extra words.\n\n"
                f"<TEXT>\n{wikitext['shortened_text']}"
            )

            if is_gemini:
                try:
                    simplified = generate_with_gemini(user_instruction)
                except Exception as e:
                    msg = str(e)
                    # If Gemini blocked the content or a server error occurred, skip the entire entry.
                    # Also remove any partially created entry from in-memory results and persist immediately.
                    if "PROHIBITED_CONTENT" in msg or "500" in msg:
                        print(f"[Skip] Skipped '{title}' due to Gemini API error: {msg[:100]}...")
                        # Remove the entry entirely so it won't be saved at all
                        try:
                            if title in title_to_entry:
                                entry_ref = title_to_entry.pop(title, None)
                                if entry_ref is not None and entry_ref in results:
                                    results.remove(entry_ref)
                            # Persist the removal immediately to ensure it is not saved later
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)
                        except OSError as se:
                            print(f"Failed to save after API error skip: {se}")
                        updated = False
                        break
                    # Otherwise, keep the existing fatal behavior
                    try:
                        filtered = [
                            r for r in results
                            if isinstance(r.get("simplified"), dict) and len(r["simplified"]) > 0
                        ]
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(filtered, f, ensure_ascii=False, indent=4)
                    except OSError as se:
                        print(f"Failed to save the file before exit: {se}")
                    print(f"[Fatal] All API keys failed. {e}")
                    sys.exit(1)
            else:
                messages = [{"role": "user", "content": user_instruction}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                try:
                    outputs = llm.generate(text, sampling_params)
                    simplified = outputs[0].outputs[0].text
                except Exception as e:
                    simplified = f"[ERROR DURING GENERATION: {e}]"

            entry["simplified"][level] = simplified
            updated = True

        # If PROHIBITED_CONTENT occurred, we broke out of the level loop without updating.
        # Continue to next entry without saving this one.
        if not updated:
            continue

        # Save only if something new was generated
        if updated:
            updated_entries_since_save += 1
            if updated_entries_since_save % 10 == 0:
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                except OSError as e:
                    print(f"Failed to save the file: {e}")

    # Flush remaining unsaved updates (if any)
    if updated_entries_since_save % 10 != 0 and updated_entries_since_save > 0:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        except OSError as e:
            print(f"Failed to save the file: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
