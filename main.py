import random
import statistics
import plotille
from rastrigin import rastrigin
import numpy as np
import matplotlib.pyplot as plt

settings = {
    "parameters": {
        "iterations": 20,
        "islands": 2,
        "agentsPerIsland": 50,
        "minRast": -5.12,
        "maxRast": 5.12,
        "minVec": 0,
        "maxVec": 1.8,
        "variance_of_vector": 0.5
    },
    "actions": [
        {
            "name": "fight",
            "reqEnergy": 0
        },
        {
            "name": "reproduce",
            "reqEnergy": 7
        },
        {
            "name": "migration",
            "reqEnergy": 9
        }
    ]
}

class Agent:
    def __init__(self, x, s):
        self.x = x
        self.s = s
        self.energy = self.evaluate()
        self.d = settings["parameters"]["variance_of_vector"]

    def evaluate(self):
        return rastrigin(self.x)

    @staticmethod
    def fight(agent_1, agent_2):
        if agent_1.energy < agent_2.energy:
            agent_2.energy = -1
        else:
            agent_1.energy = -1

    def mutate(self):
        self.s[0] *= np.exp(np.random.normal(0, self.d))
        self.s[1] *= np.exp(np.random.normal(0, self.d))

        self.x[0] += np.random.normal(0, self.s[0])
        self.x[1] += np.random.normal(0, self.s[1])

    @staticmethod
    def crossover(parent1, parent2):
        parent1_genotype = (parent1.x, parent1.s)
        parent2_genotype = (parent2.x, parent2.s)

        newborn_x = [0, 0]
        newborn_s = [0, 0]
        if random.random() < 0.5:
            newborn_x[0] = parent1_genotype[0][0]
            newborn_s[0] = parent1_genotype[1][0]
        else:
            newborn_x[0] = parent2_genotype[0][0]
            newborn_s[0] = parent2_genotype[1][0]

        if random.random() < 0.5:
            newborn_x[1] = parent1_genotype[0][1]
            newborn_s[1] = parent1_genotype[1][1]
        else:
            newborn_x[1] = parent2_genotype[0][1]
            newborn_s[1] = parent2_genotype[1][1]

        return newborn_x, newborn_s

    @staticmethod
    def reproduce(parent1, parent2):
        if random.randint(1, 2) == 1:
            newborn_x, newborn_y = Agent.crossover(parent1, parent2)
        else:
            newborn_x = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
                        random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
            newborn_y = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
                        random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]

        newborn = Agent(newborn_x, newborn_y)

        if random.randint(1, 4) == 1:
            newborn.mutate()

        return newborn

    def is_dead(self):
        return self.energy < 0


class Island:
    def __init__(self, agents, islands):
        self.agents = agents
        self.islands = islands

    def perform_actions(self):
        random.shuffle(self.agents)
        to_be_migrated = []
        newborns = []
        initial_agent_amount = len(self.agents)

        # Reproduce
        for idx, agent_1 in enumerate(self.agents):
            agent_2 = random.choice([agent_t for agent_t in self.agents if agent_t != agent_1])
            newborn = Agent.reproduce(agent_1, agent_2)
            newborns.append(newborn)
            if idx == initial_agent_amount - 1:
                break
        self.agents.extend(newborns)

        # Fight
        for idx, agent_1 in enumerate(self.agents):
            if agent_1.is_dead():
                initial_agent_amount += 1
                continue
            agent_2 = random.choice([agent_t for agent_t in self.agents if agent_t != agent_1 and not agent_t.is_dead()])
            Agent.fight(agent_1, agent_2)
            if idx == initial_agent_amount - 1:
                break

        # Remove dead
        self.remove_dead()

        # Migrate
        for agent in self.agents:
            if random.randint(1, 10) == 1:
                to_be_migrated.append(agent)
        self.migrate(to_be_migrated)

    def remove_dead(self):
        self.agents = [agent for agent in self.agents if not agent.is_dead()]

    def migrate(self, to_be_migrated):
        for agent in to_be_migrated:
            island = random.choice(self.islands)
            island.agents.append(agent)
            self.agents.remove(agent)


def prepare_input_data():
    for _ in range(settings["parameters"]["islands"]):
        x = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
             random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
        s = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
             random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]
        yield x, s, settings["parameters"]["agentsPerIsland"]


def main():
    islands = []
    output_data = {"iteration": [], "island": []}
    scatter_data = {"island": []}

    for input_data in prepare_input_data():
        island = Island([Agent(input_data[0], input_data[1]) for _ in range(input_data[2])], islands)
        islands.append(island)
        output_data["island"].append([])
        scatter_data["island"].append([])

    for island_idx in range(len(islands)):
        scatter_data["island"][island_idx].append(
            [(agent.x[0], agent.x[1], agent.energy) for agent in islands[island_idx].agents])

    for it in range(settings["parameters"]["iterations"]):
        output_data["iteration"].append(it)
        for island_idx, island in enumerate(islands):
            island.perform_actions()
            energies = [agent.energy for agent in island.agents]
            if len(energies) == 0:
                energies.append(0)
            output_data["island"][island_idx].append(statistics.mean(energies))
            scatter_data["island"][island_idx].append(
                [(agent.x[0], agent.x[1], agent.energy) for agent in islands[island_idx].agents])

    for island_idx, island in enumerate(islands):
        print(f"Final agents on island {island_idx}:")

        print("\nEnergy plot for iterations:")
        print(plotille.plot(output_data["iteration"], output_data["island"][island_idx], height=30, width=60,
                            interp="linear",
                            lc="cyan"))

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        rate1 = 0.003
        rate2 = 0.02
        rate3 = 0.003

        min_point = [-1, -1, -1]

        for series_idx, series in enumerate(scatter_data["island"][island_idx]):
            x_val = []
            y_val = []
            z_val = []

            for point in series:
                if point[2] >= 200: continue
                x_val.append(point[0])
                y_val.append(point[1])
                z_val.append(point[2])

                if min_point[2] == -1 or min_point[2] > point[2]:
                    min_point[0] = point[0]
                    min_point[1] = point[1]
                    min_point[2] = point[2]

                ax.scatter(x_val, y_val, z_val, color=[rate1 * series_idx, rate2 * series_idx, rate3 * series_idx])

        print("Minimum in: x: ", min_point[0], " y: ", min_point[1], " energy: ", min_point[2])

        plt.show()


if __name__ == "__main__":
    main()
