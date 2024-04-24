import json
import math
import random
from rastrigin import rastrigin
from sphere import sphere_function
import numpy as np
import matplotlib.pyplot as plt
import copy

fitness_function = sphere_function

numberOfIterations = 30000
numberOfAgents = 40
dimensions = 100
minRast = -5.12
maxRast = 5.12

settings = {
    "startEnergy": 1000,
    "mutation_probability": 1,
    "mutation_element_probability": 0.2,
    "crossover_probability": 0.5,
    "distribution_index": 0.2,
    "fightLossEnergy": 0.05,
    "reproduceLossEnergy": 0.3,
    "fightReqEnergy": 0,
    "reproduceReqEnergy": 1700,
    "deathThreshold":8,
    "crowdingFactor":100
}


class Agent:
    def __init__(self, x, energy=settings["startEnergy"]):
        self.x = x
        self.energy = energy
        self.fitness = fitness_function(x)

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
        # for i in range(len(x)):
        #     rand = random.random()
        #     if rand <= 1/len(x):
        #         x[i] = (
        #             minRast + (maxRast - minRast) * random.random()
        #         )
        # return x
        for i in range(len(x)):
            rand = random.random()

            if rand <= 1 / len(x):
                y = x[i]
                yl, yu = minRast, maxRast

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
                    if y < minRast:
                        y = minRast
                    if y > maxRast:
                        y = maxRast
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
            newborns = Agent.crossover(parent1, parent2)
            newborn_x1, newborn_x2 = newborns[0], newborns[1]
        else:
            newborns = Agent.crossover(parent2, parent1)
            newborn_x1, newborn_x2 = newborns[0], newborns[1]

        mutation_probability_x1 = mutation_probability_x2 = settings["mutation_probability"]

        if fitness_function(newborn_x1) < f_avg:
            mutation_probability_x1 /= 2
        else:
            mutation_probability_x1 *= 2

        if fitness_function(newborn_x2) < f_avg:
            mutation_probability_x2 /= 2
        else:
            mutation_probability_x2 *= 2

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
            if d<settings["crowdingFactor"]:
                energy =  agent_2.energy * (1-d**2/settings["crowdingFactor"]**2)
                agent_1.energy += energy
                agent_2.energy -= energy
        else:
            energy = agent_1.energy * loss_energy
            agent_1.energy -= energy
            agent_2.energy += energy
            if d<settings["crowdingFactor"]:
                energy =  agent_1.energy * (1-d**2/settings["crowdingFactor"]**2)
                agent_1.energy -= energy
                agent_2.energy += energy

        agent_1.energy = np.true_divide(np.floor(agent_1.energy * 10**settings["deathThreshold"]), 10**settings["deathThreshold"])
        agent_2.energy = np.true_divide(np.floor(agent_2.energy * 10**settings["deathThreshold"]), 10**settings["deathThreshold"])

        
    

    # @staticmethod
    # def fight(agent_1, agent_2, loss_energy):
    #     if agent_1.fitness < agent_2.fitness:
    #         energy = agent_2.energy
    #         agent_1.energy += energy
    #         agent_2.energy -= energy
    #     else:
    #         energy = agent_1.energy
    #         agent_1.energy -= energy
    #         agent_2.energy += energy

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
        req_energy = settings["fightReqEnergy"]
        loss_energy = settings["fightLossEnergy"]

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
    return [Agent([random.uniform(minRast, maxRast) for _ in range(dimensions)]) for _ in range(numberOfAgents)]


def save_to_file(output):
    settings['function'] = fitness_function.__name__
    settings['output'] = output
    try:
        with open("results.txt", 'a+') as file:
            json.dump(settings, file, indent=4)
            file.write('\n')
    except Exception as e:
        print("Error while saving results to file:", e)


def main():
    agents = generate_agents()

    emas = EMAS(agents)

    total_number_of_born, total_number_of_dead = 0, 0
    data = []

    for it in range(numberOfIterations):
        # Number of agents, born agents and dead agents
        born_num, dead_num = emas.run_iteration()
        total_number_of_born += born_num
        total_number_of_dead += dead_num
        agents_num = len(emas.agents)
        energy_sum = np.sum([agent.energy for agent in emas.agents])

        if it%50 == 0:
            print(it, agents_num)


        # Min and Max standard deviations along each dimension for agents
        vectors = np.array([agent.x for agent in emas.agents])
        std = np.std(vectors, axis=0)
        min_std = min(std)
        max_std = max(std)

        # Best agent based on its fitness
        best_agent = min(emas.agents, key=lambda agent: agent.fitness)
        lowest_energy = min(emas.agents, key=lambda agent: agent.energy).energy
        # print(it, agents_num)

        # Add data
        data.append((
            agents_num,
            born_num,
            dead_num,
            best_agent.fitness,
            np.average([agent.fitness for agent in emas.agents]),
            best_agent.energy,
            np.average([agent.energy for agent in emas.agents]),
            min_std,
            max_std,
            energy_sum
        ))
        if it%100 == 0:
            print(f'\n\n')
            print(min_std)
            # for item in data:
            #     print(item)
            # print(f"\n{it}")
            # print(f"Lowest energy: {lowest_energy}")
            # print(len(emas.agents),[agent.energy for agent in emas.agents])
            # input()

    print("Number of agents left:", len(emas.agents))
    print()
    print("Total number of born agents:", total_number_of_born)
    print("Total number of dead agents:", total_number_of_dead)
    print()

    best_agent = min(emas.agents, key=lambda agent: agent.fitness)

    for i in range(len(best_agent.x)):
        best_agent.x[i] = round(best_agent.x[i], 2)

    output = f"Minimum in {best_agent.x} equals = {best_agent.fitness:.2f} for agent with energy equals = {best_agent.energy:.2f}"
    print(output)

    iteration_data = list(range(len(data)))
    number_of_agents = [item[0] for item in data]
    number_of_born_agents = [item[1] for item in data]
    number_of_dead_agents = [item[2] for item in data]
    best_fitness = [item[3] for item in data]
    avg_fitness = [item[4] for item in data]
    best_energy = [item[5] for item in data]
    avg_energy = [item[6] for item in data]
    min_std = [item[7] for item in data]
    max_std = [item[8] for item in data]
    sum_energy = [item[9] for item in data]

    fig, ax = plt.subplots(5, 2)
    fig.set_figheight(30)
    fig.set_figwidth(20)

    ax[0, 0].plot(iteration_data, number_of_agents, marker='o', linestyle='-')
    ax[0, 0].set_title("Number of agents after each iterations")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Number of agents")
    ax[0, 0].grid()

    ax[0, 1].plot(iteration_data, sum_energy, marker='o', linestyle='-')
    ax[0, 1].set_title("Energy sum")
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Energy sum")
    ax[0, 1].grid()

    ax[1, 0].plot(iteration_data, number_of_born_agents, marker='o', linestyle='-')
    ax[1, 0].set_title("Number of born agents after each iteration")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Born")
    ax[1, 0].grid()

    ax[1, 1].plot(iteration_data, number_of_dead_agents, marker='o', linestyle='-')
    ax[1, 1].set_title("Number of dead agents after each iteration")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("Dead")
    ax[1, 1].grid()

    ax[2, 0].plot(iteration_data, best_fitness, marker='o', linestyle='-')
    ax[2, 0].set_title("Best fitness after each iteration")
    ax[2, 0].set_xlabel("Iteration")
    ax[2, 0].set_ylabel("Best fitness")
    ax[2, 0].grid()

    ax[2, 1].plot(iteration_data, avg_fitness, marker='o', linestyle='-')
    ax[2, 1].set_title("Average fitness after each iteration")
    ax[2, 1].set_xlabel("Iteration")
    ax[2, 1].set_ylabel("Avg fitness")
    ax[2, 1].grid()

    ax[3, 0].plot(iteration_data, best_energy, marker='o', linestyle='-')
    ax[3, 0].set_title("Best energy after each iteration")
    ax[3, 0].set_xlabel("Iteration")
    ax[3, 0].set_ylabel("Best energy")
    ax[3, 0].grid()

    ax[3, 1].plot(iteration_data, avg_energy, marker='o', linestyle='-')
    ax[3, 1].set_title("Average energy after each iteration")
    ax[3, 1].set_xlabel("Iteration")
    ax[3, 1].set_ylabel("Avg energy")
    ax[3, 1].grid()

    ax[4, 0].plot(iteration_data, min_std, marker='o', linestyle='-')
    ax[4, 0].set_title("Min standard deviation after each iteration")
    ax[4, 0].set_xlabel("Iteration")
    ax[4, 0].set_ylabel("min std")
    ax[4, 0].grid()

    ax[4, 1].plot(iteration_data, max_std, marker='o', linestyle='-')
    ax[4, 1].set_title("Max standard deviation after each iteration")
    ax[4, 1].set_xlabel("Iteration")
    ax[4, 1].set_ylabel("max std")
    ax[4, 1].grid()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    fig.suptitle(fitness_function.__name__ + ' minimization', fontsize=14)
    plt.show()

    save_to_file(output)


if __name__ == "__main__":
    main()
