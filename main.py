import random

settings = {
    "parameters":{
        "iterations":10,
        "islands":2,
        "agentsPerIsland":10,
        "maxStartingGenotype": 10,
        "maxStartingEnergyLvl": 15
    },
    "actions":[
        {
            "name":"fight",
            "reqEnergy":0
        },
        {
            "name":"reproduce",
            "reqEnergy":7
        },
        {
            "name":"migration",
            "reqEnergy":9
        }
    ]
}
class Agent:
    def __init__(self, genotype, energy):
        self.genotype = genotype
        self.energy = energy

    def fight(self, neighbor):
        if self.genotype > neighbor.genotype:
            energy = neighbor.energy * 0.1 # Zabieramy część energii sąsiada
            self.energy += energy
            neighbor.energy -= energy  
        else:
            energy = self.energy * 0.1 # Przekazujemy część naszej energii
            neighbor.energy += energy  
            self.energy -= energy

    def reproduce(self, neighbor, reqEnergy):
        if self.energy > reqEnergy and neighbor.energy > reqEnergy:
            child_genotype = (self.genotype + neighbor.genotype) / 2  # Średnia genotypów rodziców
            child_energy = (self.energy + neighbor.energy) / 2  # Średnia energii rodziców
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
            self.agents.remove(agent)
            island = random.choice(self.islands)
            island.agents.append(agent)
        

def main():
    islands=[]
    for _ in range(settings["parameters"]["islands"]):
        island = Island([Agent(random.randint(1, settings["parameters"]["maxStartingGenotype"]), random.randint(1, settings["parameters"]["maxStartingEnergyLvl"])) for _ in range(settings["parameters"]["agentsPerIsland"])],islands)
        islands.append(island)


    for _ in range(settings["parameters"]["iterations"]):
        for island in islands:
            island.perform_actions()
            island.perform_death()


    for i, island in enumerate(islands):
        print(f"Final agents on island {i}:")
        for agent in island.agents:
            print("Genotype:", agent.genotype, "Energy:", agent.energy)

if __name__ == "__main__":
    main()
