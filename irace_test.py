import math
import random
import numpy as np
import copy
from rastrigin import rastrigin
from sphere import sphere_function
from irace import irace

func = rastrigin
LB = [-5.12]
UB = [5.12]

DIM = 50

numberOfIterations = 1000
numberOfAgents = 40

parameters_table = '''
start_energy                 "" i (0,100)
mutation_probability         "" r (0,1)
mutation_element_probability "" r (0,1)
crossover_probability        "" r (0,1)
distribution_index           "" r (0,1)
fight_loss_energy            "" r (0,1)
reproduce_loss_energy        "" r (0,1)
fight_req_energy             "" i (0,100)
reproduce_req_energy         "" i (0,100)
death_treshold               "" i (0,10)
'''

default_values = '''
    start_energy mutation_probability mutation_element_probability crossover_probability distribution_index fight_loss_energy reproduce_loss_energy fight_req_energy reproduce_req_energy death_treshold
    100          0.5                  0.5                          0.5                   0.2                0.2               0.25                  0                0                    7
'''


class Agent:
    def __init__(self, x, settings, energy=None):
        self.x = x
        self.settings = settings

        if energy is not None:
            self.energy = energy
        else:
            self.energy = settings["start_energy"]

        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return func(self.x)

    @staticmethod
    def crossover(parent1, parent2):
        parents = [parent1, parent2]
        offspring = copy.deepcopy(parents)
        permutation_length = len(offspring[0].x)

        cross_points = sorted([random.randint(0, permutation_length) for _ in range(2)])

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
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                            except ValueError as ve:
                                print('ValueError encountered, Action skipped')
                                break
            return s1, s2

        swapped = _swap(parents[0].x, parents[1].x, cross_points)
        mapped = _map(swapped, cross_points)

        offspring[0].x, offspring[1].x = mapped

        return offspring[0].x, offspring[1].x

    @staticmethod
    def mutate(x, settings):
        for i in range(len(x)):
            rand = random.random()

            if rand <= 1 / len(x):
                y = x[i]
                yl, yu = LB[0], UB[0]

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
                    if y < LB[0]:
                        y = LB[0]
                    if y > UB[0]:
                        y = UB[0]
                x[i] = y
        return x

    @staticmethod
    def reproduce(parent1, parent2, loss_energy, f_avg, settings):
        parent1_loss = math.ceil(parent1.energy * loss_energy)
        parent1.energy -= parent1_loss

        parent2_loss = math.ceil(parent2.energy * loss_energy)
        parent2.energy -= parent2_loss

        if random.random() < settings["crossover_probability"]:
            newborns = Agent.crossover(parent1, parent2)
            newborn_x1, newborn_x2 = newborns[0], newborns[1]
        else:
            newborns = Agent.crossover(parent2, parent1)
            newborn_x1, newborn_x2 = newborns[0], newborns[1]

        mutation_probability_x1 = mutation_probability_x2 = settings["mutation_probability"]

        if func(newborn_x1) < f_avg:
            mutation_probability_x1 /= 2
        else:
            mutation_probability_x1 *= 2

        if func(newborn_x2) < f_avg:
            mutation_probability_x2 /= 2
        else:
            mutation_probability_x2 *= 2

        random_number = random.random()
        if random_number < mutation_probability_x1:
            newborn_x1 = Agent.mutate(newborn_x1, settings)
        if random_number < mutation_probability_x2:
            newborn_x2 = Agent.mutate(newborn_x2, settings)

        newborn1 = Agent(newborn_x1, settings, parent1_loss + parent2_loss)
        newborn2 = Agent(newborn_x2, settings, parent1_loss + parent2_loss)

        return newborn1 if newborn1.fitness < newborn2.fitness else newborn2

    # @staticmethod
    # def fight(agent_1, agent_2, loss_energy):
    #     if agent_1.fitness < agent_2.fitness:
    #         energy = math.ceil(max(agent_2.energy * loss_energy, settings["minFightEnergyLoss"]))
    #         agent_1.energy += energy
    #         agent_2.energy -= energy
    #     else:
    #         energy = math.ceil(max(agent_1.energy * loss_energy, settings["minFightEnergyLoss"]))
    #         agent_1.energy -= energy
    #         agent_2.energy += energy

    @staticmethod
    def fight(agent_1, agent_2, loss_energy, death_treshold):
        if agent_1.fitness < agent_2.fitness:
            energy = agent_2.energy * loss_energy
            agent_1.energy += energy
            agent_2.energy -= energy
        else:
            energy = agent_1.energy * loss_energy
            agent_1.energy -= energy
            agent_2.energy += energy
        
        agent_1.energy = np.true_divide(np.floor(agent_1.energy * 10**death_treshold), 10**death_treshold)
        agent_2.energy = np.true_divide(np.floor(agent_2.energy * 10**death_treshold), 10**death_treshold)

    def is_dead(self):
        return self.energy <= 0


class EMAS:
    def __init__(self, seed, agents, settings):
        self.seed = seed
        self.agents = agents
        self.settings = settings

    def run_iteration(self):
        random.shuffle(self.agents)

        children = self.reproduce()
        self.fight()
        self.agents.extend(children)
        self.clear()

    def reproduce(self):
        req_energy = self.settings["reproduce_req_energy"]
        loss_energy = self.settings["reproduce_loss_energy"]

        parents = []
        children = []
        for idx, parent1 in enumerate(self.agents):
            if parent1.energy > req_energy and parent1 not in parents:
                available_parents = [agent for agent in self.agents if
                                     agent != parent1 and agent.energy > req_energy and agent not in parents]
                if available_parents:
                    parent2 = random.choice(available_parents)
                    children.append(Agent.reproduce(parent1, parent2, loss_energy,
                                                    np.average([agent.fitness for agent in self.agents]), self.settings))
                    parents.extend([parent1, parent2])

        return children

    def fight(self):
        req_energy = self.settings["fight_req_energy"]
        loss_energy = self.settings["fight_loss_energy"]
        death_treshold = self.settings["death_treshold"]

        fighters = []
        for idx, agent1 in enumerate(self.agents):
            if agent1.energy > req_energy and agent1 not in fighters:
                available_fighters = [agent for agent in self.agents if
                                      agent != agent1 and agent.energy > req_energy and agent not in fighters]
                if available_fighters:
                    agent2 = random.choice(available_fighters)
                    Agent.fight(agent1, agent2, loss_energy, death_treshold)
                    fighters.extend([agent1, agent2])

    def clear(self):
        self.agents = [agent for agent in self.agents if not agent.is_dead()]


def generate_agents(settings):
    return [Agent([random.uniform(LB[0], UB[0]) for _ in range(DIM)], settings) for _ in range(numberOfAgents)]


def optimize(seed, config):
    agents = generate_agents(config)
    emas = EMAS(seed, agents, config)

    for _ in range(numberOfIterations):
        emas.run_iteration()

    best_agent = min(emas.agents, key=lambda agent: agent.fitness, default=None)
    return math.inf if best_agent is None else best_agent.fitness


def target_runner(experiment, scenario, lb=LB, ub=UB):
    s = experiment['seed']
    c = experiment['configuration']

    ret = optimize(s, c)

    return dict(cost=ret)


# These are dummy "instances", we are tuning only on a single function.
instances = np.arange(100)

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances=instances,
    maxExperiments=300,
    debugLevel=3,
    digits=5,
    parallel=1,
    logFile="")

tuner = irace(scenario, parameters_table, target_runner)
tuner.set_initial_from_str(default_values)
best_confs = tuner.run()
# Pandas DataFrame
print(best_confs)
