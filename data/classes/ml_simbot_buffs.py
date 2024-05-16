class Buff:
    def __init__(self, name, duration, damage_increase, stacks):
        self.name = name
        self.duration = duration
        self.damage_increase = damage_increase
        self.timer = duration
        self.stacks = stacks

    def tick(self):
        if self.timer > 0:
            self.timer -= 1
        return self.timer > 0


    """
    Fireball gibt einen Stack LivingFlame.
    Stacks capped bei 10.
    Nach 5 Runden ohne neuen Stack verfällt 1 Stack.
    
    LivingFlame base dmg = 5 
    + 10 dmg pro stack.
    """