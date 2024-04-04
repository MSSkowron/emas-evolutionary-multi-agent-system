import math
import random
from rastrigin import rastrigin
from sphere import sphere_function
import numpy as np
import matplotlib.pyplot as plt
import copy

settings = {
    "parameters": {
        "startEnergy": 100,
        "numberOfIterations": 40,
        "numberOfAgents": 100,
        "dimensions": 2,
        "minRast": -5.12,
        "maxRast": 5.12,
        "minFightEnergyLoss": 5,
        "mutation_probability": 0.5,
        "mutation_element_probability": 0.5,
        "crossover_probability": 0.5,
        "distribution_index": 0.2,
        "fitness_function": rastrigin
    },
    "actions": [
        {
            "name": "fight",
            "reqEnergy": 0,
            "lossEnergy": 0.1
        },
        {
            "name": "reproduce",
            "reqEnergy": 40,
            "lossEnergy": 0.1
        }
    ]
}


class Agent:
    def __init__(self, x, energy=settings["parameters"]["startEnergy"]):
        self.x = x
        self.energy = energy

    def fitness(self):
        return settings["parameters"]["fitness_function"](self.x)

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
    def mutate(x):
        for i in range(len(x)):
            rand = random.random()

            if rand <= settings["parameters"]["mutation_element_probability"]:
                y = x[i]
                yl, yu = settings["parameters"]["minRast"], settings["parameters"]["maxRast"]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (settings["parameters"]["distribution_index"] + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (
                            pow(xy, settings["parameters"]["distribution_index"] + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (
                            pow(xy, settings["parameters"]["distribution_index"] + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)
                    if y < settings["parameters"]["minRast"]:
                        y = settings["parameters"]["minRast"]
                    if y > settings["parameters"]["maxRast"]:
                        y = settings["parameters"]["maxRast"]
                x[i] = y
        return x

    @staticmethod
    def reproduce(parent1, parent2, loss_energy):
        parent1_loss = math.ceil(parent1.energy * loss_energy)
        parent1.energy -= parent1_loss

        parent2_loss = math.ceil(parent2.energy * loss_energy)
        parent2.energy -= parent2_loss

        # Possible crossover
        if random.random() < settings["parameters"]["crossover_probability"]:
            newborns = Agent.crossover(parent1, parent2)
            newborn_x1, newborn_x2 = newborns[0], newborns[1]
        else:
            newborn_x1 = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]) for _ in
                          range(settings["parameters"]["dimensions"])]
            newborn_x2 = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]) for _ in
                          range(settings["parameters"]["dimensions"])]

        # Possible mutation
        if random.random() < settings["parameters"]["mutation_probability"]:
            newborn_x1 = Agent.mutate(newborn_x1)
            newborn_x2 = Agent.mutate(newborn_x2)

        newborn1 = Agent(newborn_x1, parent1_loss + parent2_loss)
        newborn2 = Agent(newborn_x2, parent1_loss + parent2_loss)

        return newborn1 if newborn1.fitness() < newborn2.fitness() else newborn2

    @staticmethod
    def fight(agent_1, agent_2, loss_energy):
        if agent_1.fitness() < agent_2.fitness():
            energy = math.ceil(max(agent_2.energy * loss_energy, settings["parameters"]["minFightEnergyLoss"]))
            agent_1.energy += energy
            agent_2.energy -= energy
        else:
            energy = math.ceil(max(agent_1.energy * loss_energy, settings["parameters"]["minFightEnergyLoss"]))
            agent_1.energy -= energy
            agent_2.energy += energy

    def is_dead(self):
        return self.energy <= 0


class EMAS:
    def __init__(self, agents):
        self.agents = agents

    def run_iteration(self):
        random.shuffle(self.agents)

        children = self.reproduce()
        self.fight()
        self.agents.extend(children)
        dead = self.clear()

        return len(children), len(dead)

    def reproduce(self):
        reproduce_action = next(action for action in settings["actions"] if action["name"] == "reproduce")
        req_energy = reproduce_action.get("reqEnergy", 1)
        loss_energy = reproduce_action.get("lossEnergy", 0.1)

        parents = []
        children = []
        for idx, parent1 in enumerate(self.agents):
            if parent1.energy > req_energy and parent1 not in parents:
                available_parents = [agent for agent in self.agents if
                                     agent != parent1 and agent.energy > req_energy and agent not in parents]
                if available_parents:
                    parent2 = random.choice(available_parents)
                    children.append(Agent.reproduce(parent1, parent2, loss_energy))
                    parents.extend([parent1, parent2])

        return children

    def fight(self):
        fight_action = next(action for action in settings["actions"] if action["name"] == "fight")
        req_energy = fight_action.get("reqEnergy", 1)
        loss_energy = fight_action.get("lossEnergy", 0.1)

        fighters = []
        for idx, agent1 in enumerate(self.agents):
            if agent1.energy > req_energy and agent1 not in fighters:
                available_fighters = [agent for agent in self.agents if
                                      agent != agent1 and agent.energy > req_energy and agent not in fighters]
                if available_fighters:
                    agent2 = random.choice(available_fighters)
                    Agent.fight(agent1, agent2, loss_energy)
                    fighters.extend([agent1, agent2])

    def clear(self):
        dead = [agent for agent in self.agents if agent.is_dead()]
        self.agents = [agent for agent in self.agents if not agent.is_dead()]
        return dead


def generate_agents():
    return [Agent(
        [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]) for _ in
         range(settings["parameters"]["dimensions"])]) for _ in range(settings["parameters"]['numberOfAgents'])]


def main():
    agents = generate_agents()

    emas = EMAS(agents)

    total_number_of_born, total_number_of_dead = 0, 0
    data = []

    for it in range(settings["parameters"]["numberOfIterations"]):
        born_num, dead_num = emas.run_iteration()

        total_number_of_born += born_num
        total_number_of_dead += dead_num

        best_fitness, best_agent = math.inf, None
        for agent in emas.agents:
            fitness = agent.fitness()
            if fitness < best_fitness:
                best_fitness = fitness
                best_agent = agent

        if best_agent:
            data.append((best_agent.x, np.average([agent.fitness() for agent in emas.agents]), best_fitness,
                         np.average([agent.energy for agent in emas.agents]), best_agent.energy))

    best_fitness, best_agent = math.inf, None
    for agent in emas.agents:
        fitness = agent.fitness()
        if fitness < best_fitness:
            best_fitness = fitness
            best_agent = agent

    for i in range(len(best_agent.x)):
        best_agent.x[i] = round(best_agent.x[i], 2)

    print("Total number of born agents:", total_number_of_born)
    print("Total number of dead agents:", total_number_of_dead)
    print()
    print(f"Minimum in {best_agent.x} equals = {best_fitness:.2f} for agent with energy equals = {best_agent.energy:.2f}")

    iteration_data = [i + 1 for i in range(len(data))]
    avg_fitness_data = [item[1] for item in data]
    min_fitness_data = [item[2] for item in data]
    avg_energy_data = [item[3] for item in data]
    max_energy_data = [item[4] for item in data]

    fig, ax = plt.subplots(2, 2)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    ax[0, 0].plot(iteration_data, avg_fitness_data, marker='o', linestyle='-')
    ax[0, 0].set_title("Average fitness for each iteration")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Avg fitness")
    ax[0, 0].grid()

    ax[1, 0].plot(iteration_data, min_fitness_data, marker='o', linestyle='-')
    ax[1, 0].set_title("Min fitness for each iteration")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Min fitness")
    ax[1, 0].grid()

    ax[0, 1].plot(iteration_data, avg_energy_data, marker='o', linestyle='-')
    ax[0, 1].set_title("Average energy for each iteration")
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Avg energy")
    ax[0, 1].grid()

    ax[1, 1].plot(iteration_data, max_energy_data, marker='o', linestyle='-')
    ax[1, 1].set_title("Max energy for each iteration")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("Max energy")
    ax[1, 1].grid()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,  wspace=0.3, hspace=0.3)
    fig.suptitle(settings["parameters"]["fitness_function"].__name__ + ' minimization', fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
