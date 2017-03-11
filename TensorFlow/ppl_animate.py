import matplotlib.pyplot as plt
import numpy as np
import random
import math

class Sine():
    def __init__(self):
        self.period = 10000
        self.freq = 2
        self.amp = 1
        self.x = np.arange(0, 10000)
        wf = (2 * np.pi * self.freq * self.x / self.period) 
        self.y = self.amp * np.sin(wf)

    def mutate(self):
        l = 0.9
        h = 1.1
        mutated_sine = Sine()
        amp_rand = random.uniform(l,h)
        mutated_sine.amp = self.amp * amp_rand
        freq_rand = random.uniform(l,h)
        mutated_sine.freq = self.freq * freq_rand

        wf = (2 * np.pi * self.freq * self.x / self.period) 
        mutated_sine.y = self.amp * np.sin(wf)

        return mutated_sine

    def heuristical_mutate(self, max_cost, min_cost, cost):
        cost = cost / (max_cost * math.log(max_cost))

        if cost == 0:
            l = 0.9
            h = 1.1
            print ("why is cost 0?")
        else:
            l = 1 - cost
            h = 1 + cost

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
        total_runs = 0
        plt.title("Sine Wave")

        while(curr_cost > 0.005):
            num_runs = num_runs + 1
            total_runs = total_runs + 1
            next_baby = self.ugly_duck.mutate()

            duck_cost = cost_function(self.ugly_duck, self.beautiful_swan)
            baby_cost = cost_function(next_baby, self.beautiful_swan)
            if(baby_cost < duck_cost):
                self.ugly_duck = next_baby
                plt.clf()
                plt.plot(self.ugly_duck.x, self.ugly_duck.y, self.beautiful_swan.x, self.beautiful_swan.y)
                plt.pause(0.01)

                curr_cost = cost_function(self.ugly_duck, self.beautiful_swan)
            else:
                num_runs = num_runs - 1

            if(num_runs > 200 or total_runs > 10000):
                break
        print ("total runs: " + str(total_runs))
        return num_runs

    def descent_with_heuristics(self):
        curr_cost = cost_function(self.ugly_duck, self.beautiful_swan)
        num_runs = 0
        total_runs = 0
        plt.title("Sine Wave")

        max_cost = cost_function(self.ugly_duck, self.beautiful_swan)

        while(curr_cost > 0.005):
            num_runs = num_runs + 1
            total_runs = total_runs + 1
            next_baby = self.ugly_duck.heuristical_mutate(max_cost, 0, curr_cost)

            duck_cost = cost_function(self.ugly_duck, self.beautiful_swan)
            baby_cost = cost_function(next_baby, self.beautiful_swan)
            if(baby_cost < duck_cost):
                self.ugly_duck = next_baby
                plt.clf()
                plt.plot(self.ugly_duck.x, self.ugly_duck.y, self.beautiful_swan.x, self.beautiful_swan.y)
                plt.pause(0.01)

                curr_cost = cost_function(self.ugly_duck, self.beautiful_swan)
            else:
                num_runs = num_runs - 1

            if(num_runs > 200 or total_runs > 10000):
                break
        print ("total runs: " + str(total_runs))
        return num_runs

def cost_function(ugly_duck, beautiful_swan):
    freq_cost = abs(ugly_duck.get_freq() - beautiful_swan.get_freq())
    amp_cost = abs(ugly_duck.get_amp() - beautiful_swan.get_amp())

    return (freq_cost + amp_cost)


def run():

    target_sine_wave = Sine()
    target_sine_wave.freq = 5
    target_sine_wave.amp = 5
    wf = (2 * np.pi * target_sine_wave.freq * target_sine_wave.x / target_sine_wave.period) 
    target_sine_wave.y = target_sine_wave.amp * np.sin(wf)

    sin1 = Sine()

    incubator = MakeBetterBabies(sin1, target_sine_wave)
    num_runs = incubator.descent()

    best_bird = incubator.ugly_duck
    print ("best offspring from " + str(num_runs) + " children: ")
    print ("frequency: " + str(best_bird.freq))
    print ("amplitude: " + str(best_bird.amp))
    print ("with cost: " + str(cost_function(best_bird,target_sine_wave)))


    plt.plot(best_bird.x, best_bird.y)
    plt.title("Sine Wave")
    plt.show()


def run_with_heuristic():
    target_sine_wave = Sine()
    target_sine_wave.freq = 5
    target_sine_wave.amp = 5
    wf = (2 * np.pi * target_sine_wave.freq * target_sine_wave.x / target_sine_wave.period) 
    target_sine_wave.y = target_sine_wave.amp * np.sin(wf)

    sin1 = Sine()

    incubator = MakeBetterBabies(sin1, target_sine_wave)
    num_runs = incubator.descent_with_heuristics()

    best_bird = incubator.ugly_duck
    print ("best offspring from " + str(num_runs) + " children: ")
    print ("frequency: " + str(best_bird.freq))
    print ("amplitude: " + str(best_bird.amp))
    print ("with cost: " + str(cost_function(best_bird,target_sine_wave)))


    plt.plot(best_bird.x, best_bird.y)
    plt.title("Sine Wave")
    plt.show()

print ("running with normal stochastic descent")
run()
print ("running with heuristic descent")
run_with_heuristic()





