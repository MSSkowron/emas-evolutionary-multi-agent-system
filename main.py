import random


class Agent:
    def __init__(self, genotype, energy):
        self.genotype = genotype
        self.energy = energy

    def interact(self, neighbor):
        if self.genotype > neighbor.genotype:
            self.energy += neighbor.energy * 0.1  # Zabieramy część energii sąsiada
        else:
            neighbor.energy += self.energy * 0.1  # Przekazujemy część naszej energii

    def reproduce(self, neighbor):
        if self.energy > 10 and neighbor.energy > 10:
            child_genotype = (self.genotype + neighbor.genotype) / 2  # Średnia genotypów rodziców
            child_energy = (self.energy + neighbor.energy) / 2  # Średnia energii rodziców
            return Agent(child_genotype, child_energy)
        else:
            return None

    def die(self):
        return self.energy < 1


class Island:
    def __init__(self, agents):
        self.agents = agents

    def perform_interaction(self):
        for agent in self.agents:
            neighbor = random.choice(self.agents)
            agent.interact(neighbor)

    def perform_reproduction(self):
        new_agents = []
        for agent in self.agents:
            neighbor = random.choice(self.agents)
            child = agent.reproduce(neighbor)
            if child:
                new_agents.append(child)
        self.agents.extend(new_agents)

    def perform_death(self):
        self.agents = [agent for agent in self.agents if not agent.die()]

    def migrate(self, other_island):
        if len(self.agents) > 10:
            migrating_agent = random.choice(self.agents)
            self.agents.remove(migrating_agent)
            other_island.agents.append(migrating_agent)


def main():
    island1 = Island([Agent(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)])
    island2 = Island([Agent(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)])

    for _ in range(10):
        island1.perform_interaction()
        island1.perform_reproduction()
        island1.perform_death()
        island1.migrate(island2)

        island2.perform_interaction()
        island2.perform_reproduction()
        island2.perform_death()
        island2.migrate(island1)

    print("Final agents on island 1:")
    for agent in island1.agents:
        print("Genotype:", agent.genotype, "Energy:", agent.energy)

    print("Final agents on island 2:")
    for agent in island2.agents:
        print("Genotype:", agent.genotype, "Energy:", agent.energy)


if __name__ == "__main__":
    main()
