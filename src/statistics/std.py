import json
import statistics
from collections import defaultdict
import argparse

def calculate_std(json_data):    
    # 1. Define level hierarchy to identify 'easy' levels (lowest 2 for each language)
    # Note: JLPT uses N5 as the easiest and N1 as the hardest.
    EASY_LEVELS = {
        "ko": ["TOPIK Level 1", "TOPIK Level 2"],
        "zh": ["HSK 3.0 Level 1", "HSK 3.0 Level 2"],
        "en": ["CEFR A1", "CEFR A2"],
        "ja": ["JLPT N5", "JLPT N4"]
    }

    # 2. Initialize a nested dictionary to store lists of values for calculation
    # Structure: storage[language][level][metric] = [value1, value2, ...]
    storage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    samples = json_data.get("samples", [])

    # 3. Aggregate metric values from samples
    for sample in samples:
        lang = sample["language"]
        level = sample["level"]
        metrics = sample["metrics"]

        # Check if the current sample belongs to the 'easy' category
        is_easy_level = level in EASY_LEVELS.get(lang, [])

        for metric_name, value in metrics.items():
            # A. Store for the specific level (e.g., "ko" -> "TOPIK Level 4")
            storage[lang][level][metric_name].append(value)
            
            # B. Store for the language 'all' aggregate (e.g., "ko" -> "all")
            storage[lang]["all"][metric_name].append(value)
            
            # C. Store for the language 'easy' aggregate (e.g., "ko" -> "easy")
            if is_easy_level:
                storage[lang]["easy"][metric_name].append(value)

            # D. Store for the global 'average' -> 'all' aggregate
            storage["average"]["all"][metric_name].append(value)

            # E. Store for the global 'average' -> 'easy' aggregate
            if is_easy_level:
                storage["average"]["easy"][metric_name].append(value)

    # 4. Calculate Standard Deviation based on the aggregated lists
    # We will mirror the structure of the input 'summary' but store STD values.
    summary_std = {}

    # Iterate through keys expected in the summary (or just use collected keys)
    # Using collected keys ensures we cover everything found in samples.
    for lang_key, levels in storage.items():
        summary_std[lang_key] = {}
        
        for level_key, metric_dict in levels.items():
            summary_std[lang_key][level_key] = {}
            
            for metric_key, values in metric_dict.items():
                if len(values) > 1:
                    # Calculate sample standard deviation
                    std_dev = statistics.stdev(values)
                else:
                    # STD is 0 if there is only 0 or 1 sample
                    std_dev = 0.0
                
                summary_std[lang_key][level_key][metric_key] = std_dev

    # 5. Add the calculated standard deviations to the original data structure
    json_data["summary_std"] = summary_std
    
    return json_data

if __name__ == "__main__":
    # Load input JSON from argparse
    parser = argparse.ArgumentParser(description="Calculate standard deviations for metrics in JSON data.")
    parser.add_argument("--input", help="Path to the input JSON file.")
    parser.add_argument("--output", help="Path to the output JSON file.")
    args = parser.parse_args()
    with open(args.input, 'r', encoding='utf-8') as f:
        input_json_str = f.read()

    # Parse the input JSON
    data = json.loads(input_json_str)
    
    # Process the data
    result_data = calculate_std(data)
    
    # Print the result (Focusing on the newly created summary_std)
    # ensure_ascii=False is used to display Korean characters correctly if printed
    print(json.dumps(result_data["summary_std"], indent=4, ensure_ascii=False))

    # To save to a file:
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)