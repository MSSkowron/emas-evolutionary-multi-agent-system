import math
import random
from rastrigin import rastrigin
import numpy as np
import matplotlib.pyplot as plt

settings = {
    "parameters": {
        "numberOfIterations": 10,
        "numberOfAgents": 100,
        "minRast": -5.12,
        "maxRast": 5.12,
        "minVec": 0,
        "maxVec": 1.8,
        "variance_of_vector": 0.5
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
        },
        {
            "name": "migration",
            "reqEnergy": 9
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
        child_x = [parent1.x[i] if random.random() < 0.5 else parent2.x[i] for i in range(len(parent1.x))]
        child_y = [parent1.y[i] if random.random() < 0.5 else parent2.y[i] for i in range(len(parent1.y))]
        return child_x, child_y

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

        # Possible crossover (50%)
        if random.randint(1, 2) == 1:
            newborn_x, newborn_y = Agent.crossover(parent1, parent2)
        else:
            newborn_x = [random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"]),
                         random.uniform(settings["parameters"]["minRast"], settings["parameters"]["maxRast"])]
            newborn_y = [random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"]),
                         random.uniform(settings["parameters"]["minVec"], settings["parameters"]["maxVec"])]

        # Possible mutation (25%)
        if random.randint(1, 4) == 1:
            newborn_x, newborn_y = Agent.mutate(newborn_x, newborn_y)

        return Agent(newborn_x, newborn_y)

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
