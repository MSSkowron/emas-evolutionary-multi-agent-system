import json
import math
import random
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

from rastrigin import func
from rastrigin import LB, UB, funcName

# from sphere import func
# from sphere import LB, UB, funcName

# from schaffer import func
# from schaffer import LB, UB, funcName

# from schwefel import func
# from schwefel import LB, UB, funcName

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

dimensions = 100
numberOfAgents = 20
maxNumberOfFitnessEvaluations = 1000

numberOfFitnessEvaluations = 0
numberOfBornAgents, numberOfDeadAgents = 0, 0

emasIsRunning = False
data = []


def update_data():
    global data, numberOfBornAgents, numberOfDeadAgents, emasIsRunning

    if not emasIsRunning:
        return

    agents_num = len(emas.agents)
    energy_sum = np.sum([agent.energy for agent in emas.agents])

    vectors = np.array([agent.x for agent in emas.agents])
    std = np.std(vectors, axis=0)
    min_std = np.min(std)
    max_std = np.max(std)

    best_agent = min(emas.agents, key=lambda agent: agent.fitness)
    if len(data) % 100 == 0:
        print(f"Evaluation: {len(data)} fitness: {best_agent.fitness}")
    data.append((
        agents_num,
        numberOfBornAgents,
        numberOfDeadAgents,
        best_agent.fitness,
        np.mean([agent.fitness for agent in emas.agents]),
        best_agent.energy,
        np.mean([agent.energy for agent in emas.agents]),
        min_std,
        max_std,
        energy_sum
    ))


class Agent:
    def __init__(self, x, energy=settings["startEnergy"]):
        self.x = x
        self.energy = energy
        self.fitness = func(x)

        global numberOfFitnessEvaluations
        numberOfFitnessEvaluations += 1

        update_data()

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
    def mutate(x):
        for i in range(len(x)):
            rand = random.random()

            if rand <= 1 / len(x):
                y = x[i]
                yl, yu = LB, UB

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
                    if y < LB:
                        y = LB
                    if y > UB:
                        y = UB
                x[i] = y
        return x

    @staticmethod
    def reproduce(parent1, parent2, loss_energy, f_avg):
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
            newborn_x1 = Agent.mutate(newborn_x1)
        if random_number < mutation_probability_x2:
            newborn_x2 = Agent.mutate(newborn_x2)

        newborn1 = Agent(newborn_x1, parent1_loss + parent2_loss)
        newborn2 = Agent(newborn_x2, parent1_loss + parent2_loss)

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
    def __init__(self, agents):
        self.agents = agents

    def run_iteration(self):
        global numberOfBornAgents, numberOfDeadAgents

        # shuffle
        random.shuffle(self.agents)

        # reproduce
        children = self.reproduce()
        numberOfBornAgents += len(children)

        # fight
        self.fight()

        # update agents' array
        self.agents.extend(children)

        # remove dead
        dead = self.clear()
        numberOfDeadAgents += len(dead)

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
                    children.append(Agent.reproduce(parent1, parent2, loss_energy,
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


def save_to_file(file_name, output, start_time, end_time):
    settings['function'] = funcName
    settings['agents'] = numberOfAgents
    settings['dimensions'] = dimensions
    settings['output'] = output
    settings['time'] = end_time - start_time
    try:
        with open(file_name, 'a+') as file:
            json.dump(settings, file, indent=4)
            file.write('\n')
    except Exception as e:
        print("Error while saving results to file:", e)


'''
==============
main()
==============
'''

start_time = time.time()

emas = EMAS([Agent([random.uniform(LB, UB) for _ in range(dimensions)])
            for _ in range(numberOfAgents)])
emasIsRunning = True

update_data()

while numberOfFitnessEvaluations < maxNumberOfFitnessEvaluations:
    emas.run_iteration()

end_time = time.time()

print("Number of agents left:", len(emas.agents))
print()
print("Total number of fitness evaluations:", numberOfFitnessEvaluations)
print()
print("Total number of born agents:", numberOfBornAgents)
print("Total number of dead agents:", numberOfDeadAgents)
print()

best_agent = min(emas.agents, key=lambda agent: agent.fitness)

for i in range(len(best_agent.x)):
    best_agent.x[i] = round(best_agent.x[i], 2)

output = f"Minimum in {best_agent.x} equals = {best_agent.fitness:.2f} for agent with energy equals = {best_agent.energy:.2f}"
print(output)

agents_num, born_agents, dead_agents = [], [], []
best_fitness, avg_fitness = [], []
best_energy, avg_energy = [], []
std_min, std_max = [], []
energy_sum = []

for i, snapshot in enumerate(data):
    agents_num.append(snapshot[0])
    born_agents.append(snapshot[1])
    dead_agents.append(snapshot[2])
    best_fitness.append(snapshot[3])
    avg_fitness.append(snapshot[4])
    best_energy.append(snapshot[5])
    avg_energy.append(snapshot[6])
    std_min.append(snapshot[7])
    std_max.append(snapshot[8])
    energy_sum.append(snapshot[9])

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
fig.tight_layout(pad=5.0)

axes[0, 0].plot(best_fitness, label='Best Fitness')
axes[0, 0].plot(avg_fitness, label='Average Fitness')
axes[0, 0].legend()
axes[0, 0].set_title('Fitness over Time')
axes[0, 0].set_xlabel('Fitness evaluations')
axes[0, 0].set_ylabel('Fitness')

axes[0, 1].plot(best_energy, label='Best Energy')
axes[0, 1].plot(avg_energy, label='Average Energy')
axes[0, 1].legend()
axes[0, 1].set_title('Energy over Time')
axes[0, 1].set_xlabel('Fitness evaluations')
axes[0, 1].set_ylabel('Energy')

axes[1, 0].plot(agents_num, label='Number of Agents')
axes[1, 0].legend()
axes[1, 0].set_title('Agents over Time')
axes[1, 0].set_xlabel('Fitness evaluations')
axes[1, 0].set_ylabel('Number of Agents')

axes[1, 1].plot(std_min, label='Minimum Std Dev')
axes[1, 1].plot(std_max, label='Maximum Std Dev')
axes[1, 1].legend()
axes[1, 1].set_title('Diversity over Time')
axes[1, 1].set_xlabel('Fitness evaluations')
axes[1, 1].set_ylabel('Standard Deviation')

axes[2, 0].plot(born_agents, label='Born Agents')
axes[2, 0].plot(dead_agents, label='Dead Agents')
axes[2, 0].legend()
axes[2, 0].set_title('Total Born and Dead Agents over Time')
axes[2, 0].set_xlabel('Fitness evaluations')
axes[2, 0].set_ylabel('Number of Agents')

axes[2, 1].plot(energy_sum, label='Total Energy')
axes[2, 1].legend()
axes[2, 1].set_title('Total Energy over Time')
axes[2, 1].set_xlabel('Fitness evaluations')
axes[2, 1].set_ylabel('Energy')

# plt.show()
file_name = "results/"+funcName+"_"+str(time.time())
plt.savefig(file_name+".png")
save_to_file(file_name+".txt", output, start_time, end_time)
