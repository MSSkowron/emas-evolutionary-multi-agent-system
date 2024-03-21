import random
import statistics
import plotille
from rastrigin import rastrigin
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

settings = {
    "parameters": {
        "iterations": 30,
        "islands": 2,
        "agentsPerIsland": 10,
        "maxStartingGenotype": 10,
        "maxStartingEnergyLvl": 15,
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
        self.energy = rastrigin(x)
        self.x = x
        self.s = s
        self.d = settings["parameters"]["variance_of_vector"]

    def fight(self, neighbor):
        if self.energy < neighbor.energy:
            neighbor.energy = -1
        else:
            self.energy = -1

    def reproduce(self, neighbor):

        selfS = copy.copy(self.s) 
        selfS[0] *= np.exp(np.random.normal(0, self.d))
        selfS[1] *= np.exp(np.random.normal(0, self.d))

        selfX = copy.copy(self.x)
        selfX[0] += np.random.normal(0, selfS[0])
        selfX[1] += np.random.normal(0, selfS[1])

        neighborS = copy.copy(neighbor.s) 
        neighborS[0] *= np.exp(np.random.normal(0, neighbor.d))
        neighborS[1] *= np.exp(np.random.normal(0, neighbor.d))

        neighborX = copy.copy(neighbor.x)
        neighborX[0] += np.random.normal(0, neighborS[0])
        neighborX[1] += np.random.normal(0, neighborS[1])

        newborn = 0
        cross_genotype = 0
        if self.energy < neighbor.energy:
            newborn = Agent(selfX, selfS)
            cross_genotype = (neighborX, neighborS)
        else:
            newborn = Agent(neighborX, neighborS)
            cross_genotype = (selfX, selfS)
        
        if random.randint(1, 4) == 4:
            newborn.x[0] = cross_genotype[0][0]
            newborn.s[0] = cross_genotype[1][0]

        if random.randint(1, 4) == 4:
            newborn.x[1] = cross_genotype[0][1]
            newborn.s[1] = cross_genotype[1][1]
        
        return newborn

    def is_dead(self):
        return self.energy < 0
    
    def print(self):
        print("e: ", self.energy)


class Island:
    def __init__(self, agents, islands):
        self.agents = agents
        self.islands = islands

    def perform_actions(self):
        random.shuffle(self.agents)
        to_be_migrated = []
        newborns = []
        initial_agent_amount = len(self.agents)

        for idx, agent in enumerate(self.agents):
            neighbor = random.choice([ agent_t for agent_t in self.agents if agent_t != agent])
            child = agent.reproduce(neighbor)
            self.agents.append(child)
            if idx == initial_agent_amount-1:
                break

        for idx, agent in enumerate(self.agents):
            if agent.is_dead():
                initial_agent_amount+=1
                continue
            neighbor = random.choice([ agent_t for agent_t in self.agents if agent_t != agent and not agent_t.is_dead()])
            agent.fight(neighbor)
            if idx == initial_agent_amount-1:
                break

        self.perform_death()
        for agent in self.agents:
            if random.randint(0, 11) == 10:
                to_be_migrated.append(agent)
        
        self.migrate(to_be_migrated)

    def perform_death(self):
        self.agents = [agent for agent in self.agents if not agent.is_dead()]

    def migrate(self, to_be_migrated):
        for agent in to_be_migrated:
            island = random.choice(self.islands)
            island.agents.append(agent)
            self.agents.remove(agent)


def prepare_input_data():
    for _ in range(settings["parameters"]["islands"]):
        x = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]), \
                random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
        s = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]), \
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
        scatter_data["island"][island_idx].append([(agent.x[0], agent.x[1], agent.energy) for agent in islands[island_idx].agents])

    for it in range(settings["parameters"]["iterations"]):
        output_data["iteration"].append(it)
        for island_idx, island in enumerate(islands):
            island.perform_actions()
            energies = [agent.energy for agent in island.agents]
            if len(energies) == 0:
                energies.append(0)
            output_data["island"][island_idx].append(statistics.mean(energies))
            scatter_data["island"][island_idx].append([ (agent.x[0], agent.x[1], agent.energy) for agent in islands[island_idx].agents ])

    for island_idx, island in enumerate(islands):
        print(f"Final agents on island {island_idx}:")

        print("\nEnergy plot for iterations:")
        print(plotille.plot(output_data["iteration"], output_data["island"][island_idx], height=30, width=60, interp="linear",
                            lc="cyan"))

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        x_val = []
        y_val = []
        z_val = []
        rate1 = 0.003
        rate2 = 0.025
        rate3 = 0.003

        for series_idx, series in enumerate(scatter_data["island"][island_idx]):
            x_val = []
            y_val = []
            z_val = []

            for point in series:
                if point[2] >= 200: continue
                x_val.append(point[0])
                y_val.append(point[1])
                z_val.append(point[2])

                ax.scatter(x_val, y_val, z_val, color=[rate1*series_idx, rate2*series_idx, rate3*series_idx])

        plt.show()

if __name__ == "__main__":
    main()
