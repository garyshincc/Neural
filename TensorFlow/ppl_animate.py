import matplotlib.pyplot as plt
import numpy as np
import random



class Sine():
    def __init__(self):
        self.period = 10000
        self.freq = 2
        self.amp = 1
        self.x = np.arange(0, 10000)
        wf = (2 * np.pi * self.freq * self.x / self.period) 
        self.y = self.amp * np.sin(wf)

    def mutate(self):
        l = 0.8
        h = 1.2
        mutated_sine = Sine()
        amp_rand = random.uniform(l,h)
        mutated_sine.amp = self.amp * amp_rand
        freq_rand = random.uniform(l,h)
        mutated_sine.freq = self.freq * freq_rand

        wf = (2 * np.pi * self.freq * self.x / self.period) 
        mutated_sine.y = self.amp * np.sin(wf)

        return mutated_sine

    def get_freq(self):
        return self.freq

    def get_amp(self):
        return self.amp

class MakeBetterBabies():
    def __init__(self, ugly_duck, beautiful_swan):
        self.ugly_duck = ugly_duck
        self.beautiful_swan = beautiful_swan

    def get_duck(self):
        return self.ugly_duck
    def get_swan(self):
        return self.beautiful_swan


    def descent(self):
        curr_cost = cost_function(self.ugly_duck, self.beautiful_swan)
        num_runs = 0
        plt.title("Sine Wave")

        while(num_runs < 50):
            num_runs = num_runs + 1
            next_baby = self.ugly_duck.mutate()

            duck_cost = cost_function(self.ugly_duck, self.beautiful_swan)
            baby_cost = cost_function(next_baby, self.beautiful_swan)
            if(baby_cost < duck_cost):
                self.ugly_duck = next_baby
                plt.clf()
                plt.plot(self.ugly_duck.x, self.ugly_duck.y, self.beautiful_swan.x, self.beautiful_swan.y)
                plt.pause(0.1)

                curr_cost = cost_function(self.ugly_duck, self.beautiful_swan)
                print ("current cost: " + str(curr_cost))
            else:
                num_runs = num_runs - 1

            if(curr_cost < 0.03):
                break
        return num_runs

def cost_function(ugly_duck, beautiful_swan):
    freq_cost = abs(ugly_duck.get_freq() - beautiful_swan.get_freq())
    amp_cost = abs(ugly_duck.get_amp() - beautiful_swan.get_amp())

    return (freq_cost + amp_cost)



target_sine_wave = Sine()
target_sine_wave.freq = 10
target_sine_wave.amp = 5
wf = (2 * np.pi * target_sine_wave.freq * target_sine_wave.x / target_sine_wave.period) 
target_sine_wave.y = target_sine_wave.amp * np.sin(wf)


sin1 = Sine()

incubator = MakeBetterBabies(sin1, target_sine_wave)

num_runs = incubator.descent()

best_bird = incubator.ugly_duck
best_bird_freq = best_bird.freq
best_bird_amp = best_bird.amp
print ("best offspring from " + str(num_runs) + " children: ")
print ("frequency: " + str(best_bird.freq))
print ("amplitude: " + str(best_bird.amp))
print ("with cost: " + str(cost_function(best_bird,target_sine_wave)))


plt.plot(best_bird.x, best_bird.y)
plt.title("Sine Wave")
plt.show()










