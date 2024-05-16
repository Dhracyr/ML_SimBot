# damage-simbot
A local damage simulation for any RPG using a genetic algorithm and CUDA.
Depending if you have a gpu that uses CUDA (e.g. Nvidia) you can either launch `SimBot.py` for the CUDA-Version or `SimBot_no_cuda.py` if you don't have access to it.
The CUDA-Version runs about 5-6x faster, than the no CUDA one.

If you want to play yourself, you can launch `max_damage_calculator.py` and play in the console.

In the future it will use spells of your online character, read buffs/cooldowns via API and trys to deal the maximum damage possible.

With the recent 6 Spells reaching from buffs, debuffs and dots - it reaches a maximum of  
*`4127.5 of 4242.5 damage in 128 ticks, that's 97,29%`*
