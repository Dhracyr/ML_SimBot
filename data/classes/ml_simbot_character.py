class Character:
    def __init__(self, trainingsdummy):
        self.buff_damage_increase = 0.0
        self.buff_name = ""
        self.buff_active = False
        self.buff_timer = 0
        self.last_spell = None
        self.damage_done = 0
        self.trainingsdummy = trainingsdummy

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
        if self.trainingsdummy.dot_timer > 0:
            self.deal_damage(self.trainingsdummy.dot_damage)

    def deal_damage(self, damage):
        if self.buff_active:
            self.trainingsdummy.calc_damage(damage * (1 + self.buff_damage_increase))
        else:
            self.trainingsdummy.calc_damage(damage)

    def inflict_dot(self, dot_damage, duration):
        self.trainingsdummy.apply_dot(dot_damage, duration)
