import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics(data, language='average'):
    summary = data.get('summary', {})
    lang_data = summary.get(language, {})
    
    return {
        'all': lang_data.get('all', {}),
        'easy': lang_data.get('easy', {})
    }

# Define file paths
# Main result files
file1_path = 'results/evaluation/eval_Qwen3-4B-Instruct-2507-trained.json' 
file2_path = 'results/evaluation/eval_pgv_Qwen3-4B-Instruct-2507-trained.json' 

# Baseline files (for dashed lines)
baseline_wiki_path = 'results/evaluation/eval_zeroshot_Qwen3-4B-Instruct-2507.json'
baseline_pgv_path = 'results/evaluation/eval_pgv_zeroshot_Qwen3-4B-Instruct-2507.json'

# Load data
data1 = load_data(file1_path)
data2 = load_data(file2_path)
base_wiki_data = load_data(baseline_wiki_path)
base_pgv_data = load_data(baseline_pgv_path)

target_lang = 'average'
metrics1 = extract_metrics(data1, target_lang)
metrics2 = extract_metrics(data2, target_lang)
metrics_base_wiki = extract_metrics(base_wiki_data, target_lang)
metrics_base_pgv = extract_metrics(base_pgv_data, target_lang)

categories = ['vocab', 'entailment', 'coherence']

# Get raw scores
def get_scores(metrics, group):
    return [metrics[group][cat] for cat in categories]

f1_all = get_scores(metrics1, 'all')
f1_easy = get_scores(metrics1, 'easy')

f2_all = get_scores(metrics2, 'all')
f2_easy = get_scores(metrics2, 'easy')

x = np.arange(len(categories))  # label locations
width = 0.18  # width of the bars
group_gap = 0.08 # Gap between Wiki and PGV groups

fig, ax = plt.subplots(figsize=(8, 5))

# Define positions
# Wiki group (Left side)
pos_wiki_total = x - (group_gap/2) - (width * 1.5)
pos_wiki_easy = x - (group_gap/2) - (width * 0.5)

# PGV group (Right side)
pos_pgv_total = x + (group_gap/2) + (width * 0.5)
pos_pgv_easy = x + (group_gap/2) + (width * 1.5)

# Define colors
color_wiki_total = '#1f77b4'      # Dark Blue
color_wiki_easy = '#9ecae1'       # Light Blue
color_pgv_total = '#ff7f0e'       # Dark Orange
color_pgv_easy = '#fdbf6f'        # Light Orange

# Create bars
rects1 = ax.bar(pos_wiki_total, f1_all, width, label='Wiki - Total', color=color_wiki_total)
rects2 = ax.bar(pos_wiki_easy, f1_easy, width, label='Wiki - Easy', color=color_wiki_easy)
rects3 = ax.bar(pos_pgv_total, f2_all, width, label='PGV - Total', color=color_pgv_total)
rects4 = ax.bar(pos_pgv_easy, f2_easy, width, label='PGV - Easy', color=color_pgv_easy)

# Scale bar heights to 100
for rects in (rects1, rects2, rects3, rects4):
    for rect in rects:
        rect.set_height(rect.get_height() * 100)

# Add Baseline Lines (Dashed Gray)
# Draw specific baseline for each bar (All vs Easy)
for i, cat in enumerate(categories):
    if cat in ['vocab', 'entailment']:
        # Wiki - Total (All)
        val_wiki_all = metrics_base_wiki['all'][cat] * 100
        ax.hlines(y=val_wiki_all, 
                  xmin=pos_wiki_total[i] - width/2, 
                  xmax=pos_wiki_total[i] + width/2, 
                  colors='black', linestyles='--', linewidth=1.5)

        # Wiki - Easy
        val_wiki_easy = metrics_base_wiki['easy'][cat] * 100
        ax.hlines(y=val_wiki_easy, 
                  xmin=pos_wiki_easy[i] - width/2, 
                  xmax=pos_wiki_easy[i] + width/2, 
                  colors='black', linestyles='--', linewidth=1.5)
        
        # PGV - Total (All)
        val_pgv_all = metrics_base_pgv['all'][cat] * 100
        ax.hlines(y=val_pgv_all, 
                  xmin=pos_pgv_total[i] - width/2, 
                  xmax=pos_pgv_total[i] + width/2, 
                  colors='black', linestyles='--', linewidth=1.5)

        # PGV - Easy
        val_pgv_easy = metrics_base_pgv['easy'][cat] * 100
        ax.hlines(y=val_pgv_easy, 
                  xmin=pos_pgv_easy[i] - width/2, 
                  xmax=pos_pgv_easy[i] + width/2, 
                  colors='black', linestyles='--', linewidth=1.5)

# Styling
ax.set_xticks(x)
ax.set_xticklabels(['Vocab. Cover.', 'Semantic Pres.', 'Coherence'], fontsize=20)
ax.set_ylim(0, 130) # Slightly higher to fit labels
ax.set_yticks(np.arange(0, 101, 20)) # 0 to 100 only
ax.tick_params(axis='y', labelsize=20)

ax.legend(fontsize=20, loc='upper right', ncol=2)

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        label = f'{height:.0f}'
        
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=20)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.savefig('results/plots/pgv_plot.png', dpi=300)
plt.savefig('results/plots/pgv_plot.pdf', dpi=300)