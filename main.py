import math
import random
from rastrigin import rastrigin
import numpy as np
import matplotlib.pyplot as plt

settings = {
    "parameters": {
        "numberOfIterations": 10,
        "numberOfAgents": 20,
        "minRast": -5.12,
        "maxRast": 5.12,
        "minVec": 0,
        "maxVec": 1.8,
        "variance_of_vector": 0.5,
        "crossover_probability": 0.5,
        "mutation_probability": 0.25
    },
    "actions": [
        {
            "name": "fight",
            "reqEnergy": 0,
            "lossEnergy": 0.1
        },
        {
            "name": "reproduce",
            "reqEnergy": 7,
            "lossEnergy": 0.2
        }
    ]
}


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = self.evaluate()

    def evaluate(self):
        return rastrigin(self.x)

    @staticmethod
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1.x) - 1)

        child_x1 = parent1.x[:crossover_point] + parent2.x[crossover_point:]
        child_y1 = parent1.y[:crossover_point] + parent2.y[crossover_point:]

        child_x2 = parent2.x[:crossover_point] + parent1.x[crossover_point:]
        child_y2 = parent2.y[:crossover_point] + parent1.y[crossover_point:]

        return child_x1, child_y1, child_x2, child_y2

    @staticmethod
    def mutate(x, y):
        new_x, new_y = [0 for i in range(len(x))], [0 for i in range(len(y))]
        for i in range(len(y)):
            new_y[i] = y[i] * np.exp(np.random.normal(0, settings["parameters"]["variance_of_vector"]))
            new_x[i] = x[i] + np.random.normal(0, new_y[i])
        return new_x, new_y

    @staticmethod
    def reproduce(parent1, parent2, loss_energy):
        parent1.energy -= parent1.energy * loss_energy
        parent2.energy -= parent2.energy * loss_energy

        # Possible crossover
        if random.random() < settings["parameters"]["crossover_probability"]:
            newborn_x1, newborn_y1, newborn_x2, newborn_y2 = Agent.crossover(parent1, parent2)
        else:
            newborn_x1 = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
                         random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
            newborn_y1 = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
                         random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]
            newborn_x2 = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
                         random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
            newborn_y2 = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
                         random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]

        # Possible mutation
        if random.random() < settings["parameters"]["mutation_probability"]:
            newborn_x1, newborn_y1 = Agent.mutate(newborn_x1, newborn_y1)
            newborn_x2, newborn_y2 = Agent.mutate(newborn_x2, newborn_y2)

        newborn1 = Agent(newborn_x1, newborn_y1)
        newborn2 = Agent(newborn_x2, newborn_y2)

        return newborn1 if newborn1.energy > newborn2.energy else newborn2

    @staticmethod
    def fight(agent_1, agent_2, loss_energy):
        if agent_1.energy > agent_2.energy:
            energy = agent_2.energy * loss_energy
            agent_1.energy += energy
            agent_2.energy -= energy
        else:
            energy = agent_1.energy * loss_energy
            agent_1.energy -= energy
            agent_2.energy += energy

    def is_dead(self):
        return self.energy < 0


class EMAS:
    def __init__(self, agents):
        self.agents = agents

    def run_iteration(self):
        random.shuffle(self.agents)

        self.reproduce()
        self.fight()
        self.clear()

    def reproduce(self):
        reproduce_action = next(action for action in settings["actions"] if action["name"] == "reproduce")
        req_energy = reproduce_action.get("reqEnergy", 1)
        loss_energy = reproduce_action.get("lossEnergy", 0.1)

        parents = []
        children = []
        for idx, parent1 in enumerate(self.agents):
            if parent1.energy > req_energy and parent1 not in parents:
                available_parents = [agent for agent in self.agents if agent != parent1 and agent.energy > req_energy and agent not in parents]
                if available_parents:
                    parent2 = random.choice(available_parents)
                    children.append(Agent.reproduce(parent1, parent2, loss_energy))
                    parents.extend([parent1, parent2])

        self.agents.extend(children)

    def fight(self):
        fight_action = next(action for action in settings["actions"] if action["name"] == "fight")
        req_energy = fight_action.get("reqEnergy", 1)
        loss_energy = fight_action.get("lossEnergy", 0.1)

        fighters = []
        for idx, agent1 in enumerate(self.agents):
            if agent1.energy > req_energy and agent1 not in fighters:
                available_fighters = [agent for agent in self.agents if agent != agent1 and agent.energy > req_energy and agent not in fighters]
                if available_fighters:
                    agent2 = random.choice(available_fighters)
                    Agent.fight(agent1, agent2, loss_energy)
                    fighters.extend([agent1, agent2])

    def clear(self):
        self.agents = [agent for agent in self.agents if not agent.is_dead()]


def main():
    agents = [Agent(
        [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
         random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])],
        [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
         random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]
    ) for _ in range(settings["parameters"]['numberOfAgents'])]

    emas = EMAS(agents)

    data = []
    for it in range(settings["parameters"]["numberOfIterations"]):

        print("Iteration", it)

        emas.run_iteration()

        best_energy, best_agent = math.inf, None
        for agent in emas.agents:
            if agent.energy < best_energy:
                best_energy = agent.energy
                best_agent = agent

        if best_agent is not None:
            data.append((best_agent.x, best_agent.evaluate(), best_agent.energy))

    best_energy, best_value, best_agent = math.inf, math.inf, None
    for agent in emas.agents:
        if agent.energy < best_energy:
            best_energy = agent.energy
            best_value = agent.evaluate()
            best_agent = agent

    for i in range(len(best_agent.x)):
        best_agent.x[i] = round(best_agent.x[i], 2)

    print(f"Minimum in {best_agent.x} equals = {best_value:.2f} for agent with energy equals = {best_energy:.2f}")

    iteration_data = [i + 1 for i in range(len(data))]
    energy_data = [item[2] for item in data]

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_data, energy_data, marker='o', linestyle='-')
    plt.title('Best energy plot for each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best energy')
    plt.grid(True)
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
