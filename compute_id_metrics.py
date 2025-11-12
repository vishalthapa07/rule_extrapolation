#!/usr/bin/env python3
"""
Simple script to demonstrate ID R1 and ID R2 computation for L3 (aNbN) grammar.
This script shows how the metrics are calculated in rule_extrapolation/runner.py
"""

import torch
import sys
sys.path.insert(0, '/Users/puskarkafle/Documents/rule_extrapolation')

from rule_extrapolation.data import (
    generate_test_prompts,
    check_same_number_as_bs,
    check_as_before_bs,
    A_token,
    B_token,
    SOS_token,
    prompt_grammar_rules
)

def compute_id_r1_r2(grammar="aNbN", test_prompt_length=6):
    """
    Compute ID R1 and ID R2 accuracies for in-distribution prompts.
    
    ID R1: Rule 1 accuracy (same number of a's and b's)
    ID R2: Rule 2 accuracy (a's come before b's)
    """
    print(f"Computing ID R1 and ID R2 for grammar: {grammar}")
    print(f"Test prompt length: {test_prompt_length}")
    print("-" * 50)
    
    # Generate all test prompts
    test_prompts = generate_test_prompts(test_prompt_length, grammar)
    print(f"Total test prompts generated: {len(test_prompts)}")
    
    # Separate into ID and OOD prompts
    id_prompts = []
    ood_prompts = []
    
    prompt_checker = prompt_grammar_rules(grammar)
    
    for prompt in test_prompts:
        if prompt_checker(prompt):
            id_prompts.append(prompt)
        else:
            ood_prompts.append(prompt)
    
    print(f"In-distribution (ID) prompts: {len(id_prompts)}")
    print(f"Out-of-distribution (OOD) prompts: {len(ood_prompts)}")
    print("-" * 50)
    
    # For demonstration, we'll compute metrics on ID prompts
    # In the actual code, these would be model predictions
    # Here we'll just check if the prompts themselves follow the rules
    
    rule_1_results = []
    rule_2_results = []
    
    for prompt in id_prompts:
        # Rule 1: Same number of a's and b's
        rule_1 = check_same_number_as_bs(prompt)
        rule_1_results.append(rule_1)
        
        # Rule 2: a's come before b's
        rule_2 = check_as_before_bs(prompt)
        rule_2_results.append(rule_2)
    
    # Calculate accuracies
    if len(rule_1_results) > 0:
        id_r1_accuracy = sum(rule_1_results) / len(rule_1_results)
        id_r2_accuracy = sum(rule_2_results) / len(rule_2_results)
        
        print(f"\nID R1 Accuracy (same number of a's and b's): {id_r1_accuracy:.4f}")
        print(f"ID R2 Accuracy (a's before b's): {id_r2_accuracy:.4f}")
        print(f"\nID R1 correct: {sum(rule_1_results)}/{len(rule_1_results)}")
        print(f"ID R2 correct: {sum(rule_2_results)}/{len(rule_2_results)}")
    else:
        print("No ID prompts found!")
    
    return id_r1_accuracy if len(rule_1_results) > 0 else 0.0, \
           id_r2_accuracy if len(rule_2_results) > 0 else 0.0

if __name__ == "__main__":
    print("=" * 50)
    print("ID R1 and ID R2 Computation Demo")
    print("=" * 50)
    print("\nNote: This demonstrates the computation logic.")
    print("In the actual runner.py, these metrics are computed")
    print("on model predictions, not just the prompts.\n")
    
    # Compute for L3 (aNbN) grammar
    id_r1, id_r2 = compute_id_r1_r2(grammar="aNbN", test_prompt_length=6)
    
    print("\n" + "=" * 50)
    print(f"Final Results:")
    print(f"  ID R1: {id_r1:.4f}")
    print(f"  ID R2: {id_r2:.4f}")
    print("=" * 50)

