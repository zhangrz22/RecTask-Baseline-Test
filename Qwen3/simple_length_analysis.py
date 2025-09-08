#!/usr/bin/env python3
"""
Simple sequence length analysis without external dependencies
"""

import json
import statistics

def analyze_sequence_lengths():
    """Analyze the sequence lengths in our training data"""
    
    print("Loading data files...")
    
    # Load data files
    try:
        with open('../Test/Beauty/Beauty.index.json', 'r') as f:
            index_data = json.load(f)
        
        with open('../Test/Beauty/Beauty.item.json', 'r') as f:
            item_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    
    print(f"Loaded {len(index_data)} SID sequences and {len(item_data)} items")
    
    # Analyze components
    print("\n=== Component Analysis ===")
    
    # SID sequence analysis
    sid_char_lengths = []
    sid_token_counts = []
    
    for item_id, tokens in index_data.items():
        sid_sequence = ''.join(tokens)
        sid_char_lengths.append(len(sid_sequence))
        sid_token_counts.append(len(tokens))
    
    print(f"SID Sequences:")
    print(f"  - Character length: avg={statistics.mean(sid_char_lengths):.1f}, range={min(sid_char_lengths)}-{max(sid_char_lengths)}")
    print(f"  - Token count: avg={statistics.mean(sid_token_counts):.1f}, range={min(sid_token_counts)}-{max(sid_token_counts)}")
    
    # Title analysis
    title_char_lengths = []
    title_word_counts = []
    
    for item_id, item_info in item_data.items():
        title = item_info.get('title', '')
        if title:
            title_char_lengths.append(len(title))
            title_word_counts.append(len(title.split()))
    
    print(f"\nTitles:")
    print(f"  - Character length: avg={statistics.mean(title_char_lengths):.1f}, range={min(title_char_lengths)}-{max(title_char_lengths)}")
    print(f"  - Word count: avg={statistics.mean(title_word_counts):.1f}, range={min(title_word_counts)}-{max(title_word_counts)}")
    print(f"  - 90th percentile char length: {sorted(title_char_lengths)[int(len(title_char_lengths) * 0.9)]}")
    print(f"  - 95th percentile char length: {sorted(title_char_lengths)[int(len(title_char_lengths) * 0.95)]}")
    
    # Training text analysis
    print("\n=== Training Text Analysis ===")
    
    # Prompt templates (simplified set)
    templates = [
        "{sid} represents {title}",
        "{sid} is {title}",
        "What is {sid}? It is {title}",
        "Q: What does {sid} represent? A: {title}",
        "User: What is {sid}?\nAssistant: {sid} represents {title}",
    ]
    
    all_training_lengths = []
    
    # Sample first 100 items for analysis
    sample_items = list(index_data.keys())[:100]
    
    for item_id in sample_items:
        if item_id not in item_data or not item_data[item_id].get('title'):
            continue
            
        sid_tokens = index_data[item_id]
        sid_sequence = ''.join(sid_tokens)
        title = item_data[item_id]['title']
        
        for template in templates:
            if '{sid}' in template and '{title}' in template:
                training_text = template.format(sid=sid_sequence, title=title)
                all_training_lengths.append(len(training_text))
    
    if all_training_lengths:
        sorted_lengths = sorted(all_training_lengths)
        
        print(f"Training Text Character Lengths (sample of {len(all_training_lengths)}):")
        print(f"  - Average: {statistics.mean(all_training_lengths):.1f} characters")
        print(f"  - Range: {min(all_training_lengths)} - {max(all_training_lengths)} characters")
        print(f"  - Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            idx = int(len(sorted_lengths) * p / 100)
            if idx >= len(sorted_lengths):
                idx = len(sorted_lengths) - 1
            print(f"    {p}%: {sorted_lengths[idx]} chars")
    
    # Token estimation (rough)
    print("\n=== Token Length Estimation ===")
    print("(Rough estimation: ~4 characters per token for English text)")
    
    if all_training_lengths:
        # Estimate tokens (rough: English text ~4 chars/token, but our SID tokens are longer)
        estimated_tokens = []
        for char_len in all_training_lengths:
            # SID tokens are much longer, so use different estimation
            # Assume: SID part ~6 tokens, text part ~char_len/4 tokens
            estimated_tok = 6 + char_len // 4  
            estimated_tokens.append(estimated_tok)
        
        sorted_tokens = sorted(estimated_tokens)
        
        print(f"Estimated Token Lengths:")
        print(f"  - Average: {statistics.mean(estimated_tokens):.1f} tokens")
        print(f"  - Range: {min(estimated_tokens)} - {max(estimated_tokens)} tokens")
        print(f"  - Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            idx = int(len(sorted_tokens) * p / 100)
            if idx >= len(sorted_tokens):
                idx = len(sorted_tokens) - 1
            print(f"    {p}%: {sorted_tokens[idx]} tokens")
        
        # Recommendations
        print(f"\n=== Recommendations ===")
        
        p95_tokens = sorted_tokens[int(len(sorted_tokens) * 0.95)]
        p90_tokens = sorted_tokens[int(len(sorted_tokens) * 0.90)]
        
        if p95_tokens <= 128:
            recommended = 256
        elif p95_tokens <= 256:
            recommended = 512  
        elif p95_tokens <= 512:
            recommended = 768
        else:
            recommended = 1024
        
        coverage = len([x for x in estimated_tokens if x <= recommended]) / len(estimated_tokens) * 100
        
        print(f"Recommended max_seq_length: {recommended}")
        print(f"This covers {coverage:.1f}% of training samples")
        
        print(f"\nAlternative options:")
        for seq_len in [128, 256, 512, 768, 1024]:
            if seq_len != recommended:
                alt_coverage = len([x for x in estimated_tokens if x <= seq_len]) / len(estimated_tokens) * 100
                print(f"  max_seq_length={seq_len}: covers {alt_coverage:.1f}% of samples")
        
        return recommended
    
    return 512  # Default fallback

def main():
    recommended = analyze_sequence_lengths()
    
    print(f"\n" + "="*50)
    print(f"FINAL RECOMMENDATION:")
    print(f"Update your run_training.sh:")
    print(f"--max_seq_length {recommended} \\")
    print(f"="*50)

if __name__ == "__main__":
    main()