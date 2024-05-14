import os
import numpy as np

from data.classes.ml_simbot_runsim import *

from matplotlib import pyplot as plt

from data.methods.adapt_mutation_rate import adapt_mutation_rate
from data.methods.draw_plot_all_gen import *


def run_simulation():

    def initialize_population(pop_size, action_space, sequence_length):
        return [np.random.randint(action_space, size=sequence_length)
                for _ in range(pop_size)]

    """
    def evaluate_population(ep_env, population):
        fitness_scores = []
        for solution in population:
            fitness = evaluate_solution(ep_env, solution)[0]
            fitness_scores.append((solution, fitness))
        return fitness_scores
    """

    def evaluate_solution(es_env, solution):
        spell_map = {
            0: 'Fireball',
            1: 'Frostbolt',
            2: 'BloodMoonCrescent',
            3: 'Blaze',
            4: 'ScorchDot',
            5: 'Combustion'
        }
        for name_action in solution:
            chosen_spell = spell_map[name_action]
            es_env.step(chosen_spell)

        damage_done_with_solution = es_env.training_dummy.damage_taken
        es_env.reset()  # Has a return value of a numpy array if we need it

        return damage_done_with_solution

    def crossover(parent1, parent2):
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def tournament_selection(ts_env, population, k=3):
        idxes = [np.random.randint(len(population)) for _ in range(k)]
        selected = [population[idx] for idx in idxes]
        selected_fitness = [evaluate_solution(ts_env, sol) for sol in selected]

        # Find the index of the solution with the highest total_reward manually
        max_fitness = -float('inf')  # Assumes fitness can't be lower than this
        best_index = 0
        for i, fitness in enumerate(selected_fitness):
            if fitness > max_fitness:  # Assuming the first element is total_reward
                max_fitness = fitness
                best_index = i

        return selected[best_index]

    def mutate(solution, mutation_rate, action_space):
        for i in range(len(solution)):
            if np.random.rand() < mutation_rate:
                solution[i] = np.random.randint(action_space)
        return solution

    def reproduce(ga_env, mutation_rate, new_population, pop_size, population, action_space_n, best_damage, generations_without_improvement, saved_damage_peak):
        while len(new_population) < pop_size:
            # Random of top 10%
            # index1, index2 = np.random.choice(len(top_indices), 2, replace=False)
            # parent1, parent2 = population[index1], population[index2]

            # Alter mutation_rate
            if saved_damage_peak == 0:
                saved_damage_peak = best_damage
            if best_damage == saved_damage_peak:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0
                saved_damage_peak = best_damage

            # Adapt mutation_rate if there are too many generations without improvement
            mutation_rate = adapt_mutation_rate(mutation_rate, generations_without_improvement, global_max_mutation_rate, global_min_mutation_rate)

            # Tournament Selection
            parent1 = tournament_selection(ga_env, population, global_tournament_k_amount)
            parent2 = tournament_selection(ga_env, population, global_tournament_k_amount)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, action_space_n)
            child2 = mutate(child2, mutation_rate, action_space_n)

            new_population.extend([child1, child2])
        return new_population, saved_damage_peak, generations_without_improvement

    def genetic_algorithm(ga_env, pop_size, generations, sequence_length, mutation_rate):
        action_space_n = ga_env.get_action_space_n()

        # Initialize population
        population = initialize_population(pop_size, action_space_n, sequence_length)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # This is specific to certain OS environments
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        list_best_damages = []
        list_generations = []

        generations_without_improvement = 0
        saved_damage_peak = 0.0

        ax.axhline(y=GLOBAL_CURRENT_RECORD, color='g', linestyle='--', linewidth=1, label='Current record')
        ax.axhline(y=GLOBAL_MAX_DAMAGE, color='r', linestyle='-', linewidth=1, label='Max Damage')

        line1, = ax.plot(list_generations, list_best_damages, linestyle='-', color='b')

        for generation in range(generations):
            # Evaluate all solutions in the population via RunSim()
            all_damage_as_list = [evaluate_solution(ga_env, sol) for sol in population]
            all_damage_as_list = np.array(all_damage_as_list, dtype=np.float32)
            max_damage = np.max(all_damage_as_list)

            # Print the damage that the generation did
            print(f"Generation {generation}: Max Damage {max_damage} of {GLOBAL_MAX_DAMAGE}, that's {max_damage/GLOBAL_MAX_DAMAGE*100:.2f}%")

            # Set up plot for live updating
            list_best_damages.append(max_damage)
            list_generations.append(generation)
            # Draw live-plot
            draw_plot_all_gen(line1, ax, fig, list_best_damages, list_generations, global_generations, False)
            plt.pause(0.01)
            # Save generation in list
            list_all_solutions.append(population)

            # Create the next generation
            new_population = []
            population, saved_damage_peak, generations_without_improvement = reproduce(ga_env, mutation_rate, new_population, pop_size, population, action_space_n, max_damage, generations_without_improvement, saved_damage_peak)

        plt.ioff()
        plt.show()
        return population, list_best_damages, list_generations

    env = RunSim()

    pop_size = global_pop_size
    generations = global_generations
    sequence_length = GLOBAL_MAX_TICKS

    best_population, list_best_damages, list_generations = genetic_algorithm(env, pop_size, generations, sequence_length, start_population_mutation_rate)
    best_solution = list_all_solutions[-1]

    # print("Best Performing Solution:")
    print("All Performing Solution:", best_population[-1])
    # print("Sequence of recent actions (spells):", best_solution)
    # print("Sequence of best actions (spells):", max(list_all_solutions, key=lambda x: x[1]))
    # print(list_all_solutions)


# duration
global_generations = 500

# parameter for cross-entropy
global_pop_size = 50  # 50
global_population_top_n_index = 0.1
start_population_mutation_rate = 0.01  # 0.01
global_tournament_k_amount = int(global_pop_size/10)  # 10% of pop_size
global_max_mutation_rate = 0.015  # 0.015
global_min_mutation_rate = 0.005  # 0.005

# TODO: Cross-over-rate?
# TODO: Reward Function that punishes similarity
# TODO: Diversity Checks
# TODO: New Spell: Fireball gives a stack of "flaming", each stack increases the damage of the new spell by 15%, stackable for 20 Stacks


if __name__ == "__main__":
    run_simulation()


# TODO: Parameteroptimierung, da Overfittung bei 95%
# TODO: Plotten von Generationen in Farben

# TODO: Parameter durch Cross-Entropy versuchen
# TODO: Datenbank auslagern?

"""
6 Stages of Success:
    1. Done!        Stop casting spells, that are on cooldown
    2. Done!        Stop casting Frostbolt, because its useless
    3. Done!        Stop casting DoT, when its already active
    4. Done!        Casting Combustion and BloodMoon on Cooldown, because it does most damage
    5. Done!        Alternating between Fireball and Blaze
    6. Oftentimes!  Waiting for BloodMoon cooldown when buff-cd is going to expire soon
"""