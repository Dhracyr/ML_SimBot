from data.global_variables import *


def draw_plot_all_gen(line, ax, fig, list_best_damage, list_generation, global_generations, animate=False):

    ax.set_xlabel('Generation')
    ax.set_ylabel('Total Damage')
    ax.set_xlim(0, global_generations)  # Use the number of generations for the x-axis limit
    ax.set_ylim(0, GLOBAL_MAX_DAMAGE * 1.1)  # Set maximum possible damage
    if not animate:
        line.set_data(list_generation, list_best_damage)
        fig.canvas.draw()
        fig.canvas.flush_events()

    else:
        line.set_data(list_generation, list_best_damage)
        ax.relim()
        ax.autoscale_view()

        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()


list_all_solutions = []
