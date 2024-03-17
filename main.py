import random
import statistics
import plotille

settings = {
    "parameters": {
        "iterations": 10,
        "no_islands": 2,
        "population_size_per_island": 10,
        "genotype_length": 4,
        "action_reproduce_energy_loss": 0.01,
        "action_fight_energy_loss": 0.05,
        "fight_energy_loss": 0.1,
        "newborn_energy_boost": 0.1,
    },
    "actions": [
        {"name": "fight", "reqEnergy": 0},
        {"name": "reproduce", "reqEnergy": 7},
        {"name": "migration", "reqEnergy": 9},
    ],
}


def crossover(parent1_genotype, parent2_genotype):
    crossover_point = random.randint(1, len(parent1_genotype) - 2)

    child1_genotype = (
        parent1_genotype[:crossover_point] + parent2_genotype[crossover_point:]
    )
    child2_genotype = (
        parent2_genotype[:crossover_point] + parent1_genotype[crossover_point:]
    )

    return child1_genotype, child2_genotype


def mutate(genotype):
    mutate_point = random.randint(0, len(genotype) - 1)
    genotype[mutate_point] = 0 if genotype[mutate_point] == 1 else 0


def invert(genotype):
    invert_start_point = random.randint(0, len(genotype) - 1)
    invert_end_point = random.randint(invert_start_point, len(genotype) - 1)

    for i in range(invert_start_point, invert_end_point + 1):
        genotype[i] = 0 if genotype[i] == 1 else 1


class Agent:
    def __init__(self, genotype):
        self.genotype = genotype
        self.energy = self.evaluate()

    def evaluate(self):
        value = 0
        for bit in self.genotype:
            value = (value << 1) | bit
        return value

    def fight(self, neighbor):
        fight_energy_loss = settings["parameters"]["fight_energy_loss"]
        action_fight_energy_loss = settings["parameters"]["action_fight_energy_loss"]

        self.energy -= self.energy * action_fight_energy_loss
        neighbor.energy -= neighbor.energy * action_fight_energy_loss

        if self.energy > neighbor.energy:
            energy = neighbor.energy * fight_energy_loss
            self.energy += energy
            neighbor.energy -= energy
        else:
            energy = self.energy * fight_energy_loss
            self.energy -= energy
            neighbor.energy += energy

    def reproduce(self, neighbor, req_energy):
        action_reproduce_energy_loss = settings["parameters"][
            "action_reproduce_energy_loss"
        ]

        self.energy -= self.energy * action_reproduce_energy_loss
        neighbor.energy -= neighbor.energy * action_reproduce_energy_loss

        if self.energy > req_energy and neighbor.energy > req_energy:
            child1_genotype, child2_genotype = crossover(
                self.genotype, neighbor.genotype
            )
            mutate(child1_genotype)
            mutate(child2_genotype)
            invert(child1_genotype)
            invert(child2_genotype)

            child1, child2 = Agent(child1_genotype), Agent(child2_genotype)

            energy_boost = settings["parameters"]["newborn_energy_boost"]
            child1.energy = (self.energy + neighbor.energy + child1.energy) / 3
            child1.energy += child1.energy * energy_boost

            child2.energy = (self.energy + neighbor.energy + child2.energy) / 3
            child2.energy += child2.energy * energy_boost

            return child1 if child1.energy > child2.energy else child2
        else:
            return None

    def dead(self):
        return self.energy < ((2 ** settings["parameters"]["genotype_length"]) - 1) // 2


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
        self.agents = [agent for agent in self.agents if not agent.dead()]

    def migrate(self, to_be_migrated):
        for agent in to_be_migrated:
            island = random.choice(self.islands)
            island.agents.append(agent)
            agent.energy -= agent.energy * 0.1
            self.agents.remove(agent)


def main():
    islands = []
    output_data = {"iteration": [], "island": []}

    for _ in range(settings["parameters"]["no_islands"]):
        island = Island(
            [
                Agent(
                    [
                        random.randint(0, 1)
                        for _ in range(settings["parameters"]["genotype_length"])
                    ]
                )
                for _ in range(settings["parameters"]["population_size_per_island"])
            ],
            islands,
        )
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

        # print("\nGenotype and energy plot:")
        # genotype_y = [agent.genotype for agent in island.agents]
        # energy_x = [agent.energy for agent in island.agents]
        # print(
        #     plotille.plot(
        #         energy_x, genotype_y, height=10, width=60, interp="linear", lc="yellow"
        #     )
        # )

        print("\nEnergy plot for iterations:")
        print(
            plotille.plot(
                output_data["iteration"],
                output_data["island"][i],
                height=30,
                width=60,
                interp="linear",
                lc="cyan",
            )
        )


if __name__ == "__main__":
    main()
