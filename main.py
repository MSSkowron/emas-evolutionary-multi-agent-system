import random
import statistics
import plotille

settings = {
    "parameters": {
        "iterations": 10,
        "islands": 2,
        "agentsPerIsland": 10,
        "maxStartingGenotype": 10,
        "maxStartingEnergyLvl": 15
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
    def __init__(self, genotype, energy):
        self.genotype = genotype
        self.energy = energy

    def fight(self, neighbor):
        # energy loss due to fight
        self.energy -= self.energy * 0.05
        neighbor.energy -= neighbor.energy * 0.05

        if self.genotype > neighbor.genotype:
            energy = neighbor.energy * 0.1
            self.energy += energy
            neighbor.energy -= energy
        else:
            energy = self.energy * 0.1
            neighbor.energy += energy
            self.energy -= energy

    def reproduce(self, neighbor, req_energy):
        # energy loss due to reproduction
        self.energy -= self.energy * 0.1
        neighbor.energy -= neighbor.energy * 0.1

        if self.energy > req_energy and neighbor.energy > req_energy:
            child_genotype = (self.genotype + neighbor.genotype) / 2
            child_energy = (self.energy + neighbor.energy) / 2
            child_energy += child_energy * 0.2  # energy boost for newborn
            return Agent(child_genotype, child_energy)
        else:
            return None

    def die(self):
        return self.energy < 1


class Island:
    def __init__(self, agents, islands):
        self.agents = agents
        self.islands = islands

    def perform_actions(self):
        random.shuffle(self.agents)
        to_be_migrated = []

        for agent in self.agents:
            action = random.choice(settings["actions"])
            if action["reqEnergy"] <= agent.energy:
                if action["name"] == "fight":
                    neighbor = random.choice(self.agents)
                    agent.fight(neighbor)
                elif action["name"] == "reproduce":
                    neighbor = random.choice(self.agents)
                    child = agent.reproduce(neighbor, action["reqEnergy"])
                    if child:
                        self.agents.append(child)
                elif action["name"] == "migrate":
                    to_be_migrated.append(agent)

        self.migrate(to_be_migrated)

    def perform_death(self):
        self.agents = [agent for agent in self.agents if not agent.die()]

    def migrate(self, to_be_migrated):
        for agent in to_be_migrated:
            island = random.choice(self.islands)
            island.agents.append(agent)
            agent.energy -= agent.energy * 0.1
            self.agents.remove(agent)


def prepare_input_data():
    for _ in range(settings["parameters"]["islands"]):
        max_starting_genotype = random.randint(1, settings["parameters"]["maxStartingGenotype"])
        max_starting_energy_lvl = random.randint(1, settings["parameters"]["maxStartingEnergyLvl"])
        yield max_starting_genotype, max_starting_energy_lvl, settings["parameters"]["agentsPerIsland"]


def main():
    islands = []
    output_data = {"iteration": [], "island": []}

    for input_data in prepare_input_data():
        island = Island([Agent(input_data[0], input_data[1]) for _ in range(input_data[2])], islands)
        islands.append(island)
        output_data["island"].append([])

    for it in range(settings["parameters"]["iterations"]):
        output_data["iteration"].append(it)
        for idx, island in enumerate(islands):
            island.perform_actions()
            island.perform_death()
            energies = [agent.energy for agent in island.agents]
            if len(energies) == 0:
                energies.append(0)
            output_data["island"][idx].append(statistics.mean(energies))

    for i, island in enumerate(islands):
        print(f"Final agents on island {i}:")
        for agent in island.agents:
            print("Genotype:", agent.genotype, "Energy:", agent.energy)

        print("\nGenotype and energy plot:")
        genotype_y = [agent.genotype for agent in island.agents]
        energy_x = [agent.energy for agent in island.agents]
        print(plotille.plot(energy_x, genotype_y, height=10, width=60, interp="linear", lc="yellow"))

        print("\nEnergy plot for iterations:")
        print(plotille.plot(output_data["iteration"], output_data["island"][i], height=30, width=60, interp="linear",
                            lc="cyan"))


if __name__ == "__main__":
    main()
