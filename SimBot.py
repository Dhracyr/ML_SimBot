import os
import numba
import numpy as np

from data.classes.ml_simbot_runsim import *

from data.global_variables import *

from matplotlib import pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from data.methods.adapt_mutation_rate import adapt_mutation_rate
from data.methods.draw_plot_all_gen import *


def run_simulation():

    def initialize_population_cuda(action_space_n, pop_size, sequence_length):
        # Initialize population
        output_array = cuda.device_array((pop_size, sequence_length),
                                         dtype=np.float32)
        # Calculate grid dimensions for cuda
        threads_per_block = 64
        blocks_per_grid = (pop_size + threads_per_block - 1) // threads_per_block
        # Random workaround for cuda
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=GLOBAL_STARTING_SEED)
        # Launch Kernel
        # population = initialize_population(pop_size, ga_env.action_space, sequence_length)
        initialize_population_kernel[blocks_per_grid, threads_per_block](rng_states, action_space_n, sequence_length,
                                                                         output_array)
        population = output_array.copy_to_host()
        return blocks_per_grid, population, rng_states, threads_per_block

    @cuda.jit
    def initialize_population_kernel(rng_states, action_space_n, sequence_length, cuda_output):
        idx = cuda.grid(1)
        if idx < cuda_output.shape[0]:
            for i in range(sequence_length):
                random_float = xoroshiro128p_uniform_float32(rng_states, idx)
                cuda_output[idx, i] = int(random_float * action_space_n)

    """
    def evaluate_population(population):
        fitness_scores = []
        for solution in population:
            fitness = evaluate_solution(solution)[0]
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
        es_env.training_dummy.damage_taken = 0

        return damage_done_with_solution

    @cuda.jit(device=True)
    def crossover_kernel(child1, child2, parent1, parent2, rng_states, idx):
        crossover_point = int(xoroshiro128p_uniform_float32(rng_states, idx) * len(parent1))
        # Single-point crossover

        # Create child1 by copying parts of parent1 and parent2
        for i in range(crossover_point):
            child1[i] = parent1[i]
        for i in range(crossover_point, len(parent1)):
            child1[i] = parent2[i]

        # Create child2 by copying parts of parent2 and parent1
        for i in range(crossover_point):
            child2[i] = parent2[i]
        for i in range(crossover_point, len(parent1)):
            child2[i] = parent1[i]

        return child1, child2

    @cuda.jit(device=True)
    def tournament_selection_kernel(population, all_damage_as_list, rng_states, idx):
        indices = cuda.local.array(shape=(global_tournament_k_amount,), dtype=numba.int32)
        for i in range(global_tournament_k_amount):
            random_idx = int(xoroshiro128p_uniform_float32(rng_states, idx) * len(population))
            indices[i] = random_idx

        best_idx = indices[0]
        for j in range(1, global_tournament_k_amount):
            if all_damage_as_list[indices[j]] > all_damage_as_list[best_idx]:
                best_idx = indices[j]
        return best_idx

    @cuda.jit(device=True)
    def mutate_kernel(solution, mutation_rate, action_space_n, rng_states, idx):
        for i in range(len(solution)):
            if xoroshiro128p_uniform_float32(rng_states, idx + i) < mutation_rate:
                new_action = int(xoroshiro128p_uniform_float32(rng_states, idx + i) * action_space_n)
                solution[i] = new_action % action_space_n
        return solution

    def reproduce(generation, generations_without_improvement,
                  mutation_rate, pop_size, population_input_device, saved_damage_peak,
                  action_space_n, blocks_per_grid, threads_per_block, best_damage,
                  all_damage_as_list, rng_states):

        # Alter mutation_rate
        if saved_damage_peak == 0:
            saved_damage_peak = best_damage
        if best_damage == saved_damage_peak:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0
            saved_damage_peak = best_damage
        # Adapt mutation_rate if there are too many generations without improvement
        mutation_rate = adapt_mutation_rate(mutation_rate, generations_without_improvement, global_max_mutation_rate,
                                            global_min_mutation_rate)
        # Print the damage that the generation did
        print(
            f"Generation {generation}: Max Damage {best_damage} of {GLOBAL_MAX_DAMAGE}, that's {best_damage / GLOBAL_MAX_DAMAGE * 100:.2f}%")

        # Create the next generation
        # Create new_population for cuda
        output_population = cuda.device_array((pop_size, GLOBAL_MAX_TICKS),
                                              dtype=np.float32)

        reproduce_kernel[blocks_per_grid, threads_per_block](population_input_device,
                                                             output_population,
                                                             mutation_rate,
                                                             action_space_n,
                                                             all_damage_as_list,
                                                             rng_states)
        new_population = population_input_device.copy_to_host()

        return new_population, saved_damage_peak, generations_without_improvement

    @cuda.jit()
    def reproduce_kernel(population, output_population, mutation_rate, action_space_n, all_damage_as_list, rng_states):
        idx = cuda.grid(1)
        if idx < population.shape[0] // 2:
            # Tournament Selection
            idx_parent1 = tournament_selection_kernel(population, all_damage_as_list, rng_states, idx)
            idx_parent2 = tournament_selection_kernel(population, all_damage_as_list, rng_states, idx)

            parent1 = population[idx_parent1]  # ERROR MUST BE HERE... WHAT DOES TOURNAMENT_S RETURN??
            parent2 = population[idx_parent2]

            # Allocate space for children
            child1 = cuda.local.array(shape=(128,), dtype=numba.float32)
            child2 = cuda.local.array(shape=(128,), dtype=numba.float32)

            crossover_kernel(child1, child2, parent1, parent2, rng_states, idx)
            child1 = mutate_kernel(child1, mutation_rate, action_space_n, rng_states, idx)
            child2 = mutate_kernel(child2, mutation_rate, action_space_n, rng_states, idx)

            for i in range(128):
                population[idx * 2, i] = child1[i]
                if (idx * 2 + 1) < population.shape[0]:
                    population[idx * 2 + 1, i] = child2[i]

    list_all_solutions = []

    def genetic_algorithm(ga_env, pop_size, generations, sequence_length, mutation_rate):
        # Get action space for cuda
        action_space_n = ga_env.get_action_space_n()

        blocks_per_grid, population, rng_states, threads_per_block = initialize_population_cuda(action_space_n,
                                                                                                 pop_size,
                                                                                                 sequence_length)

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # This is specific to certain OS environments
        plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        list_best_damages = []
        list_generations = []
        generations_without_improvement = 0
        saved_damage_peak = 0.0

        fig, ax = plt.subplots()
        line1, = ax.plot(list_generations, list_best_damages, linestyle='-', color='b')
        ax.axhline(y=GLOBAL_CURRENT_RECORD, color='g', linestyle='--', linewidth=1, label='Current record')
        ax.axhline(y=GLOBAL_MAX_DAMAGE, color='r', linestyle='-', linewidth=1, label='Max Damage')

        # Reproduce every generation
        for generation in range(generations):
            # Evaluate all solutions in the population via RunSim()
            all_damage_as_list = [evaluate_solution(ga_env, sol) for sol in population]
            all_damage_as_list = np.array(all_damage_as_list, dtype=np.float32)
            max_damage = np.max(all_damage_as_list)

            # Set up plot for live updating
            list_best_damages.append(max_damage)
            list_generations.append(generation)
            # Draw live-plot
            draw_plot_all_gen(line1, ax, fig, list_best_damages, list_generations, global_generations, False)
            plt.pause(0.01)
            # Save generation in list
            list_all_solutions.append(population)

            # Create a new population
            population_device = cuda.device_array_like(population)
            new_population, saved_damage_peak, generations_without_improvement = reproduce(generation,
                                                                                           generations_without_improvement,
                                                                                           mutation_rate, pop_size,
                                                                                           population_device,
                                                                                           saved_damage_peak,
                                                                                           action_space_n,
                                                                                           blocks_per_grid,
                                                                                           threads_per_block,
                                                                                           max_damage,
                                                                                           all_damage_as_list,
                                                                                           rng_states)

        plt.ioff()
        plt.show()
        return list_all_solutions

    env = RunSim()

    pop_size = global_pop_size
    generations = global_generations
    sequence_length = GLOBAL_MAX_TICKS

    best_population = genetic_algorithm(env, pop_size, generations, sequence_length, start_population_mutation_rate)
    best_solution = max(list_all_solutions)

    # print("Best Performing Solution:")
    print("All Performing Solution:", best_population)
    print("Sequence of recent actions (spells):", best_solution)
    print("Sequence of best actions (spells):", max(list_all_solutions, key=lambda x: x[1]))
    # print(list_all_solutions)


# duration
global_generations = 1000

# parameter for cross-entropy
global_pop_size = 50  # 50
global_population_top_n_index = 0.1
start_population_mutation_rate = 0.01  # 0.01
global_tournament_k_amount = 3  # 10% of pop_size probably
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
