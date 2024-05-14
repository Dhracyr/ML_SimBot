def adapt_mutation_rate(current_rate, generations_without_improvement, max_rate=0.05, min_rate=0.01):
    if generations_without_improvement > 10:  # No improvement for 10 generations
        new_rate = min(current_rate * 1.3, max_rate)
        #  print("No improvement since", generations_without_improvement, "mutation_rate is getting raised to", new_rate)
    else:
        new_rate = max(current_rate * 0.95, min_rate)
        #  print("Improvement since", generations_without_improvement, "mutation_rate is getting lowered to", new_rate)
    return new_rate
