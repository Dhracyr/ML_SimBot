import random


class Spell:
    def __init__(self, name, cooldown, damage):
        self.name = name
        self.cooldown = cooldown
        self.current_cooldown = 0
        self.damage = damage

    def cast(self):
        if self.current_cooldown == 0:
            self.current_cooldown = self.cooldown
            print(f"Casting {self.name}, setting cooldown to {self.cooldown}")
            return True
        print(f"Cannot cast {self.name}, cooldown remaining: {self.current_cooldown}")
        return False

    def cooldown_tick(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1


class Fireball(Spell):
    def __init__(self, name, cooldown, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.damage_taken += self.damage


class Frostbolt(Spell):
    def __init__(self, name, cooldown, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.damage_taken += self.damage


class Blaze(Spell):
    def __init__(self, name, cooldown, damage, training_dummy, character):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy
        self.character = character

    def cast(self):
        if super().cast():
            if self.character.last_spell == "Fireball":
                applied_damage = self.damage * 5
                self.training_dummy.damage_taken += applied_damage
                print(f"Blaze cast after Fireball, damage applied: {applied_damage}")
            else:
                self.training_dummy.damage_taken += self.damage
                print(f"Blaze cast without Fireball, damage applied: {self.damage}")
            return True
        return False


class ScorchDot(Spell):
    def __init__(self, name, cooldown, duration, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.duration = duration
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.apply_dot(self.damage, self.duration)
            return True
        return False


class Combustion(Spell):
    def __init__(self, name, cooldown, duration, damage_increase, character):
        super().__init__(name, cooldown, duration)
        self.duration = duration
        self.character = character
        self.damage_increase = damage_increase

    def cast(self):
        if super().cast():
            self.character.activate_buff(self.name, self.duration, self.damage_increase)
            return True
        return False


class TrainingDummy:

    def __init__(self):
        self.dot_damage = 0
        self.dot_timer = 0
        self.damage_taken = 0

    def apply_dot(self, dot_damage, duration):
        self.dot_timer = duration
        self.dot_damage = dot_damage
        self.damage_taken += dot_damage

    def dot_damage_calculation(self):
        if self.dot_timer > 0:
            self.damage_taken += self.dot_damage

    def tick(self):
        # Check Dot-Timer
        if self.dot_timer > 0:
            self.dot_timer -= 1
            self.dot_damage_calculation()


class Character:
    def __init__(self):
        self.buff_damage_increase = None
        self.buff_name = None
        self.buff_active = False
        self.buff_timer = 0
        self.last_spell = None

    def activate_buff(self, name, duration, damage_increase):
        self.buff_active = True
        self.buff_timer = duration
        self.buff_name = name
        self.buff_damage_increase = damage_increase

    # Check Buff-Timer
    def tick(self):
        if self.buff_timer > 0:
            self.buff_timer -= 1
            if self.buff_timer == 0:
                self.buff_active = False


class RunSim:
    def __init__(self):
        self.character = Character()
        self.training_dummy = TrainingDummy()
        self.spells = {
            'Fireball': Fireball('Fireball', 0, 10, self.training_dummy),        # Name, Cooldown, Damage
            'Frostbolt': Frostbolt('Frostbolt', 0, 3, self.training_dummy),      # Name, Cooldown, Damage
            'Blaze': Blaze('Blaze', 0, 5, self.training_dummy, self.character),  # Name, Cooldown, Damage
            'DoT': ScorchDot('DoT', 0, 20, 5, self.training_dummy),              # Name, Cooldown, Duration, Damage
            'Buff': Combustion('Buff', 60, 25, 0.5, self.character)         # Name, Cooldown, Duration, Damage_increase
        }

    """
    def reset(self):
        self.character = Character()
        for spell in self.spells.values():
            spell.current_cooldown = 0
        return self.get_state()
    """
    def step(self, action_name):
        # Check last_casted_spell
        spell = self.spells[action_name]
        if spell.cast():
            self.character.last_spell = spell.name

        # Check Dot-Spell/Buff Tick
        self.training_dummy.tick()
        self.character.tick()

        # Check Cooldown Tick
        for spell in self.spells.values():
            spell.cooldown_tick()

        # Check action used
        # spell = self.spells[action]

        # Create output for visualisation
        self.render(spell)

    def render(self, action):
        print(f"Total Damage: {self.training_dummy.damage_taken} damage")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Remaining DoT Duration: {self.training_dummy.dot_timer}")
        print(f"Remaining Buff Duration: {self.character.buff_timer}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Spell cast: {action.name}")
        print(f"Spell cast last time: {self.character.last_spell}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")


if __name__ == "__main__":
    env = RunSim()
    # env.reset()

    for _ in range(70):  # Simulate 60 Ticks/Seconds
        possible_actions = list(env.spells.keys())
        action_name_main = random.choice(possible_actions)
        env.step(action_name_main)
