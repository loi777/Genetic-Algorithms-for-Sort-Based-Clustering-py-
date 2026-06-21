from debugg import debug

import random
import pandas as pd
import pandas.core.series as row

from pathlib import Path
import shutil

from typing import List

from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import use
use('Agg')

from cluster import *
from sorter import *


#===============================================#===============================================


INDIVIDUALS = 50
ELITISM = 5
GENERATIONS = 40

MUTATION_RATE = 0.20        # percentage of mutation
PARAMETER_LIMIT = 255       # limite da mutaçao de parametros
THRESHOLD_MAX = 3.5       # limite da mutação de threshold
THRESHOLD_MIN = 2.2

#===============================================#===============================================



class Individual:
    def __init__(self, parameter_size: int):
        self.parameters: List[int] = [0] * parameter_size
        self.threshold = 0
        self.fitness = 0
        self.clusters = []
        self.centroids = []
        self.csize = 0

    #--------------------------------------
        
    def calc_cluster_data(self, list):
        self.clusters = gen_cluster(list, self.threshold)     # acquire clusters
        self.csize = get_cluster_amount(self.clusters)
        self.centroids = gen_cluster_centroid(list, self.clusters)
        

    def calc_fitness(self, list):
        debug(f"\033[32m\t\tChecking next fitness!\033[0m")

        if (self.csize <= 3):    self.fitness = cluster_fitness(list, self.clusters, self.centroids)
        else :              self.fitness = cluster_fitness(list, self.clusters, self.centroids) + (len(list)*self.csize)

    #--------------------------------------

    def mutation_reset(self):
        self.threshold = random.uniform(THRESHOLD_MIN, THRESHOLD_MAX)

        for i in range(len(self.parameters)):
            self.parameters[i] = random.randint(0, PARAMETER_LIMIT)


    def mutation_evolve(self, other: "Individual"):
        mutate_parameter = int(MUTATION_RATE*PARAMETER_LIMIT)
        mutate_threshold = MUTATION_RATE*(THRESHOLD_MAX-THRESHOLD_MIN)

        #--

        self.threshold = other.threshold + (random.uniform(0, mutate_threshold*2) - mutate_threshold)
        self.threshold = max(THRESHOLD_MIN, min(self.threshold, THRESHOLD_MAX))

        for i in range(len(self.parameters)):
            self.parameters[i] = other.parameters[i] + (random.randint(0, mutate_parameter*2) - mutate_parameter)
            self.parameters[i] = max(0, min(self.parameters[i], PARAMETER_LIMIT))


#===============================================#===============================================


class Generation:
    def __init__(self, parameter_size: int):
        self.individuals = [
            Individual(parameter_size)
            for _ in range(INDIVIDUALS)
        ]
        self.age = 0

    #--------------------------------------
        
    def reset_population(self):
        debug("\033[34mReseting population\033[0m")
        for i in range(len(self.individuals)):
            self.individuals[i].mutation_reset()
            debug(f"\033[34mParameters of element[{i}]: {self.individuals[i].parameters}\033[0m")

    #--------------------------------------
            
    def calc_generation_fitness(self, list):
        for i in range(len(self.individuals)):
            debug(f"Parameters of element[{i}]: {self.individuals[i].parameters}")
            myorder(list, self.individuals[i].parameters)   # order the list according to this individual
            self.individuals[i].calc_cluster_data(list)     # get the clusters from this list
            self.individuals[i].calc_fitness(list)          # get the fitness from these clusters
            debug(f"Fitness of element[{i}]: {self.individuals[i].fitness}\n")

            
    # calculates fitness and order generation according to it
    def order_generation(self):

        self.individuals.sort(key=lambda x: x.fitness)
        for i in range(len(self.individuals)):
            debug(f"\033[31mGeracao ordenada index[{i}] fitness {self.individuals[i].fitness}\033[0m")

    #--------------------------------------
            
    # evolve 1 step of generation
    def evolve_generation(self, list: List):
        self.age += 1
        if (GENERATIONS == self.age): print(f"Generations reached the end!"); exit

        #--

        self.calc_generation_fitness(list)
        self.order_generation()

        #--

        # Only copy Elitism once and the rest is randomn
        # Do not affect the Top-Elitism
        for index in range(ELITISM):
            self.individuals[index+ELITISM].mutation_evolve(self.individuals[index%ELITISM])

        # Do not affect the Top-Elitism
        for index in range(len(self.individuals) - (ELITISM*2)):
            self.individuals[index+(ELITISM*2)].mutation_reset()

    #--------------------------------------
            
    # Run the entire loop for the GA process
    def run_GA(self, list):
        print(f"INICIANDO CODIGO GENETICO")
        while(self.age <= GENERATIONS):
            print(f"\t\tGERAÇÂO {self.age}")
            self.evolve_generation(list)

    #--------------------------------------
            
    def select_best_fitness(self) -> Individual:
        return self.individuals[0]






#===============================================#===============================================



def create_folder(path):
    if path.exists():
        shutil.rmtree(path)
    
    path.mkdir(exist_ok=True)


#===============================================#===============================================



def main():
    print("AG executing, getting input tabular dataset from /data")

    all_dataset_folders = [p for p in Path("data").iterdir() if p.is_dir()]

    #--

    # For each data folder:
    for dataset_folder in all_dataset_folders:
        print(f"\n\nFound data folder {dataset_folder.name}")

        # readie
        create_folder(Path(f"./results/{dataset_folder.name}"))

        # Find the .data file of this folder
        data_file = next(dataset_folder.glob("*.data"))
        if (data_file == None):
            print("Dataset not found, error")   # Error handler
            continue
        print(f"Dataset found as {data_file.name}")

        #--

        dataFrame = pd.read_csv(data_file)
        # turning string names into a numerical value for each unique name
        for col in dataFrame.select_dtypes(include="object"):
            dataFrame[col] = pd.factorize(dataFrame[col])[0]

        # Acquire a useful list instead of a pandas thingy
        dataset = dataFrame.to_numpy().tolist()
        dataset = normalize_list(dataset) # List is required to be normalized!!!

        #--
        # Start AG
        
        GA = Generation(len(dataset[0]))
        GA.reset_population()
        GA.run_GA(dataset)

        # after running GA completely
        best_individual = GA.select_best_fitness()
        myorder(dataset, best_individual.parameters)    # or else clusters from this individual will not make sense

        #--
        # Print graph in results
        print_clusters(dataset, dataset_folder.name, best_individual.clusters, best_individual.fitness, best_individual.threshold)

    


#===============================================#===============================================
main()