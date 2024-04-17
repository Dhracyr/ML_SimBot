import random


class Spell:
    def __init__(self, name, cooldown, damage):
        self.name = name
        self.cooldown = cooldown
        self.current_cooldown = 0
        self.damage = damage

    def cast(self, character):
        if self.current_cooldown == 0:
            self.effect(character)
            self.current_cooldown = self.cooldown
            return True
        return False

    def effect(self, character):
        character.deal_damage(self.damage)

    def cooldown_tick(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1


class BlazeSpell(Spell):
    def effect(self, character):
        if character.last_spell == "Fireball":
            character.deal_damage(self.damage * 5) # Higher damage if after Fireball
        else:
            character.deal_damage(self.damage)  


class BuffSpell(Spell):
    def __init__(self, name, cooldown, duration):
        super().__init__(name, cooldown, 0)
        self.duration = duration

    def effect(self, character):
        character.activate_buff(self.duration)


class DotSpell(Spell):
    def effect(self, character):
        character.apply_dot(self.damage, 20)  # Assuming a 20-second duration


class Character:
    def __init__(self):
        self.buff_active = False
        self.buff_timer = 0
        self.dot_timer = 0
        self.dot_damage = 0
        self.total_damage = 0
        self.last_spell = None

    def deal_damage(self, damage):
        if self.buff_active:
            damage *= 1.3
        self.total_damage += damage

    def activate_buff(self, duration):
        self.buff_active = True
        self.buff_timer = duration

    def apply_dot(self, damage, duration):
        self.dot_timer = duration
        self.dot_damage = damage

    def tick(self):
        if self.buff_timer > 0:
            self.buff_timer -= 1
            if self.buff_timer == 0:
                self.buff_active = False
        if self.dot_timer > 0:
            self.dot_timer -= 1
            self.deal_damage(self.dot_damage)


class WowSimEnv:
    def __init__(self):
        self.character = Character()
        self.spells = {
            'Fireball': Spell('Fireball', 1, 10),
            'Blaze': BlazeSpell('Blaze', 1, 5),
            'DoT': DotSpell('DoT', 20, 5),
            'Buff': BuffSpell('Buff', 60, 0)
        }

    def reset(self):
        self.character = Character()
        for spell in self.spells.values():
            spell.current_cooldown = 0
        return self.get_state()

    def get_state(self):
        # Define the state based on character and spells
        pass

    def step(self, action):
        spell = self.spells[action]
        if spell.cast(self.character):
            self.character.last_spell = spell.name
        self.character.tick()
        for spell in self.spells.values():
            spell.cooldown_tick()
        self.render()

    def render(self):
        # Improved render method to display current environment state neatly
        print(f"Total Damage: {self.character.total_damage} damage")
        for name, spell in self.spells.items():
            print(f"{name} Cooldown: {spell.current_cooldown}s")


def main():
    env = WowSimEnv()
    env.reset()

    for _ in range(120):  # Simulate for 120 ticks, roughly equivalent to 120 seconds
        possible_actions = list(env.spells.keys())
        chosen_action = random.choice(possible_actions)
        env.step(chosen_action)


if __name__ == "__main__":
    main()
