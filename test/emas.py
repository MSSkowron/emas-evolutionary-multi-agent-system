import json
import math
import random
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

from rastrigin import rastrigin
from rastrigin import LB as rastrigin_LB
from rastrigin import UB as rastrigin_UB

from sphere import sphere
from sphere import LB as sphere_LB
from sphere import UB as sphere_UB

from schwefel import schwefel
from schwefel import LB as schwefel_LB
from schwefel import UB as schwefel_UB

from schaffer import schaffer
from schaffer import LB as schaffer_LB
from schaffer import UB as schaffer_UB

settings = {
    "startEnergy": 1000,
    "mutation_probability": 1,
    "crossover_probability": 0.5,
    "distribution_index": 0.2,
    "fightLossEnergy": 0.05,
    "reproduceLossEnergy": 0.3,
    "reproduceReqEnergy": 1700,
    "deathThreshold": 8,
    "crowdingFactor": 1000
}


class Agent:
    def __init__(self, x, emas, energy=settings["startEnergy"]):
        self.x = x
        self.energy = energy
        self.emas = emas
        self.fitness = emas.function(x)

        emas.numberOfFitnessEvaluations += 1
        emas.update_data()

    @staticmethod
    def crossover(parent1, parent2):
        parents = [parent1, parent2]
        offspring = copy.deepcopy(parents)
        permutation_length = len(offspring[0].x)

        cross_points = sorted(
            [random.randint(0, permutation_length) for _ in range(2)])

        def _repeated(element, collection):
            c = 0
            for e in collection:
                if e == element:
                    c += 1
            return c > 1

        def _swap(data_a, data_b, cross_points):
            c1, c2 = cross_points
            new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
            new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
            return new_a, new_b

        def _map(swapped, cross_points):
            n = len(swapped[0])
            c1, c2 = cross_points
            s1, s2 = swapped
            map_ = s1[c1:c2], s2[c1:c2]
            for i_chromosome in range(n):
                if not c1 < i_chromosome < c2:
                    for i_son in range(2):
                        while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                            try:
                                map_index = map_[i_son].index(
                                    swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[
                                    1 - i_son][map_index]
                            except ValueError as ve:
                                print('ValueError encountered, Action skipped')
                                break
            return s1, s2

        swapped = _swap(parents[0].x, parents[1].x, cross_points)
        mapped = _map(swapped, cross_points)

        offspring[0].x, offspring[1].x = mapped

        return offspring[0].x, offspring[1].x

    @staticmethod
    def mutate(x, lowerBound, upperBound):
        for i in range(len(x)):
            rand = random.random()

            if rand <= 1 / len(x):
                y = x[i]
                yl, yu = lowerBound, upperBound

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (settings["distribution_index"] + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (
                            pow(xy, settings["distribution_index"] + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (
                            pow(xy, settings["distribution_index"] + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)
                    if y < lowerBound:
                        y = lowerBound
                    if y > upperBound:
                        y = upperBound
                x[i] = y
        return x

    @staticmethod
    def reproduce(emas, parent1, parent2, loss_energy, f_avg):
        parent1_loss = math.ceil(parent1.energy * loss_energy)
        parent1.energy -= parent1_loss

        parent2_loss = math.ceil(parent2.energy * loss_energy)
        parent2.energy -= parent2_loss

        # Possible crossover
        if random.random() < settings["crossover_probability"]:
            newborn_x1, newborn_x2 = Agent.crossover(parent1, parent2)
        else:
            newborn_x1, newborn_x2 = Agent.crossover(parent2, parent1)

        mutation_probability_x1 = mutation_probability_x2 = settings["mutation_probability"]

        # if func(newborn_x1) < f_avg:
        #     mutation_probability_x1 /= 2
        # else:
        #     mutation_probability_x1 *= 2

        # if func(newborn_x2) < f_avg:
        #     mutation_probability_x2 /= 2
        # else:
        #     mutation_probability_x2 *= 2

        random_number = random.random()
        if random_number < mutation_probability_x1:
            newborn_x1 = Agent.mutate(
                newborn_x1, emas.lowerBound, emas.upperBound)
        if random_number < mutation_probability_x2:
            newborn_x2 = Agent.mutate(
                newborn_x2, emas.lowerBound, emas.upperBound)

        newborn1 = Agent(newborn_x1, emas, parent1_loss + parent2_loss)
        newborn2 = Agent(newborn_x2, emas, parent1_loss + parent2_loss)

        return newborn1 if newborn1.fitness < newborn2.fitness else newborn2

    @staticmethod
    def fight(agent_1, agent_2, loss_energy):
        d = np.sum(np.abs(np.array(agent_1.x) - np.array(agent_2.x)))
        if agent_1.fitness < agent_2.fitness:
            energy = agent_2.energy * loss_energy
            agent_1.energy += energy
            agent_2.energy -= energy
            if d < settings["crowdingFactor"]:
                energy = agent_2.energy * \
                    (1-d**2/settings["crowdingFactor"]**2)
                agent_1.energy += energy
                agent_2.energy -= energy
        else:
            energy = agent_1.energy * loss_energy
            agent_1.energy -= energy
            agent_2.energy += energy
            if d < settings["crowdingFactor"]:
                energy = agent_1.energy * \
                    (1-d**2/settings["crowdingFactor"]**2)
                agent_1.energy -= energy
                agent_2.energy += energy

        agent_1.energy = np.true_divide(np.floor(
            agent_1.energy * 10**settings["deathThreshold"]), 10**settings["deathThreshold"])
        agent_2.energy = np.true_divide(np.floor(
            agent_2.energy * 10**settings["deathThreshold"]), 10**settings["deathThreshold"])

    def is_dead(self):
        return self.energy <= 0


class EMAS:
    def __init__(self, function, lowerBound, upperBound):
        self.function = function
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        self.agents = []
        self.numberOfFitnessEvaluations = 0
        self.emasIsRunning = False
        self.data = [[], []]

        self.best_fitness = float('inf')

    def setAgents(self, agents):
        self.agents = agents

    def run_iteration(self):

        # shuffle
        random.shuffle(self.agents)

        # reproduce
        children = self.reproduce()

        # fight
        self.fight()

        # update agents' array
        self.agents.extend(children)

        # remove dead
        dead = self.clear()

    def reproduce(self):
        req_energy = settings["reproduceReqEnergy"]
        loss_energy = settings["reproduceLossEnergy"]

        parents = []
        children = []
        for idx, parent1 in enumerate(self.agents):
            if parent1.energy > req_energy and parent1 not in parents:
                available_parents = [agent for agent in self.agents if
                                     agent != parent1 and agent.energy > req_energy and agent not in parents]
                if available_parents:
                    parent2 = random.choice(available_parents)
                    children.append(Agent.reproduce(self, parent1, parent2, loss_energy,
                                                    np.average([agent.fitness for agent in self.agents])))
                    parents.extend([parent1, parent2])

        return children

    def fight(self):
        loss_energy = settings["fightLossEnergy"]

        fighters = []
        for idx, agent1 in enumerate(self.agents):
            if agent1 not in fighters:
                available_fighters = [agent for agent in self.agents if
                                      agent != agent1 and agent not in fighters]
                if available_fighters:
                    agent2 = random.choice(available_fighters)
                    Agent.fight(agent1, agent2, loss_energy)
                    fighters.extend([agent1, agent2])

    def clear(self):
        dead = [agent for agent in self.agents if agent.is_dead()]
        self.agents = [agent for agent in self.agents if not agent.is_dead()]
        return dead

    def update_data(self):
        if not self.emasIsRunning:
            return

        best_agent = min(self.agents, key=lambda agent: agent.fitness)
        self.best_fitness = best_agent.fitness
        if self.numberOfFitnessEvaluations % 100 == 0:
            self.data[0].append(self.numberOfFitnessEvaluations)
            self.data[1].append(best_agent.fitness)


def run(dimensions, function, lowerBound, upperBound, numberOfAgents, maxNumberOfFitnessEvaluations):
    emas = EMAS(function, lowerBound, upperBound)
    agents = [Agent([random.uniform(lowerBound, upperBound) for _ in range(dimensions)], emas=emas)
              for _ in range(numberOfAgents)]
    emas.setAgents(agents)
    emas.emasIsRunning = True

    emas.update_data()

    last_best_fitness = 0
    best_fitness_change_it = 0
    while emas.numberOfFitnessEvaluations < maxNumberOfFitnessEvaluations:
        emas.run_iteration()
        if emas.best_fitness == last_best_fitness:
            if best_fitness_change_it > 100:
                print("Nothing changed in 100 iterations")
                break
            best_fitness_change_it += 1
        else:
            last_best_fitness = emas.best_fitness
            best_fitness_change_it = 0

    best_agent = min(emas.agents, key=lambda agent: agent.fitness)

    for i in range(len(best_agent.x)):
        best_agent.x[i] = round(best_agent.x[i], 2)

    return emas.data


if __name__ == "__main__":
    print(run(100, schaffer, schaffer_LB, schaffer_UB, 20, 5000))
