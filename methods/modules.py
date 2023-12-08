# for importing modules from parent directory
import os

# for time
import time

# for data manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt

# for fuzzy logic
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# for statistics
from sklearn.metrics import mean_squared_error, r2_score

# for genetic algorithm
import random


def create_automf(data, labels, num_mf):
    """
    Create num_mf memebership functions equally spaced with the custom labels.

    :param data: universe of the variable
    :param labels: labels of the mf
    :param num_mf: number of mf
    """

    # check if labels length is equal to num_mf if not fun error
    if len(labels) != num_mf:
        raise ValueError("labels length is not equal to num_mf")

    # create the universe
    universe = data.universe

    # i need to divide the universe in num_mf mf with the labels
    ranges = []
    for i in range(num_mf):
        ranges.append(universe[0] + i * (universe[-1] - universe[0]) / (num_mf - 1))

    # create the mf according to the labels
    mf = []
    for i in range(num_mf):
        if i == 0:
            mf.append([ranges[i], ranges[i], ranges[i + 1]])
        elif i == num_mf - 1:
            mf.append([ranges[i - 1], ranges[i], ranges[i]])
        else:
            mf.append([ranges[i - 1], ranges[i], ranges[i + 1]])

    # add the labels to the mf
    for label in labels:
        mean_val = mf[labels.index(label)][1]
        sigma_val = (mf[labels.index(label)][2] - mf[labels.index(label)][0]) / 10
        data[label] = fuzz.gaussmf(data.universe, mean_val, sigma_val)
        # data[label] = fuzz.trimf(data.universe, mf[labels.index(label)])


def create_individual(labels, number):
    """
    Create a random individual

    :param labels: custom labels for the linguistic variables
    :param number: size of the matrix (individual is a square matrix)
    :return: individual matrix of combinations of labels
    """
    individual = []
    for i in range(number):
        sub_individual = []
        for j in range(number):
            sub_individual.append(random.choice(labels))
        individual.append(sub_individual)

    return individual


def create_population(labels, number):
    """
    Create a population of random individuals

    :param labels: custom labels for the linguistic variables
    :param number: number of individuals
    :return: population of individuals
    """
    population = []
    for _ in range(number):
        population.append(create_individual(labels, len(labels)))
        # population.append(create_individual(labels, 5))

    return population


def create_rules(ctrl_temp, ctrl_hum, ctrl_pm10, labels, labels_predefined):
    """
    Create the rules for the fuzzy system

    :param ctrl_temp: antecedent for the temperature
    :param ctrl_hum: antecedent for the humidity
    :param ctrl_pm10: consequent for the pm10
    :param labels: custom labels for the linguistic variables
    :param labels_predefined: labels given by the genetic algorithm
    :return: rules for the fuzzy system
    """
    rules = []
    counter_i = 0
    for label_temp in labels:
        counter_j = 0
        for label_rh in labels:
            rules.append(
                ctrl.Rule(
                    ctrl_temp[label_temp] & ctrl_hum[label_rh],
                    ctrl_pm10[labels_predefined[counter_i][counter_j]],
                )
            )
            counter_j += 1
        counter_i += 1

    return rules


def calculate_fitness(population, temp, rh, pm10, control_vars, verbose=0):
    """
    The fitness function will be calculated with the MSE

    :param population: population of the genetic algorithm
    :param temp: temperature data
    :param rh: relative humidity data
    :param pm10: pm10 data
    :param control_vars: control variables for the fuzzy system (temp_ctrl, rh_ctrl, pm10_ctrl, labels)
    :param verbose: if 1 will print the fitness
    :return: fitness of the population
    """
    fitness = []
    # =================================================================
    # ============================= LABELS ============================
    # =================================================================
    labels = control_vars[3]

    # =================================================================
    # =================== ANTECEDENTS & CONSEQUENTS ===================
    # =================================================================
    # this can be ommited here because the control variables are already created

    # =================================================================
    # ====================== MEMBERSHIP FUNCTIONS =====================
    # =================================================================
    temp_ctrl, rh_ctrl, pm10_ctrl = control_vars[0], control_vars[1], control_vars[2]

    # =================================================================
    # =========================== EVALUATE ============================
    # =================================================================
    for labels_predefined in population:
        # get the rules from the population
        rules = create_rules(temp_ctrl, rh_ctrl, pm10_ctrl, labels, labels_predefined)

        # create the fuzzy system
        system = ctrl.ControlSystem(rules)
        # simulation = ctrl.ControlSystemSimulation(system, flush_after_run=1000, clip_to_bounds=True)
        simulation = ctrl.ControlSystemSimulation(system, flush_after_run=5000 + 1)

        # porcentage to train vs test
        train_p = 0.5

        # the limits to train is 0 to train_p*len(df)
        limit_2_train = int(train_p * len(temp))
        # limit_2_train = 1500

        limit = limit_2_train
        results = np.zeros(limit)
        for i in range(limit):
            simulation.input["temp"] = temp[i]
            simulation.input["rh"] = rh[i]

            # compute the fuzzy system
            simulation.compute()

            # get the result
            out = simulation.output["pm10"]

            # append the result to the results array
            results[i] = out

        # calculate the MSE
        mse = mean_squared_error(pm10[:limit], results)
        # r2 = r2_score(pm10[:limit], results)
        # r2 = (r2 - 1) ** 2

        # append the mse to the fitness array
        fitness.append(mse)
        # fitness.append(r2)

    return fitness


def select_parents(population, fitness, number=2):
    """
    Select the parents based on the fitness

    :param population: population of the genetic algorithm
    :param fitness: fitness of the population
    :param number: number of parents to select
    :return: parents selected
    """
    # get the indexes of the best fitness
    parents_index = np.argpartition(fitness, 1)[0:number]

    # get the parents
    parents = []
    for index in parents_index:
        parents.append(population[index])

    return parents


def get_worsts(population, fitness, number=2):
    """
    Select the worsts individuals based on the fitness

    :param population: population of the genetic algorithm
    :param fitness: fitness of the population
    :param number: number of worsts to select
    :return: worsts selected
    """
    # get the indexes of the best fitness
    worsts_index = np.argpartition(fitness, -2)[-number:]

    return worsts_index


def make_crossover(parents, number):
    """
    Simple crossing between parents

    :param parents: parents to cross
    :param number: size of the matrix
    :return: children created
    """

    # separate the parents
    parent1, parent2 = parents

    # create the children
    child1 = []
    child2 = []
    children = []

    # the mixing will bi [[alta baja alta baja alta], [baja alta baja alta baja], ...]
    for i in range(number):
        sub1 = []
        sub2 = []
        for j in range(number):
            if j % 2 == 0:
                sub1.append(parent1[i][j])
                sub2.append(parent2[i][j])
            else:
                sub1.append(parent2[i][j])
                sub2.append(parent1[i][j])
        child1.append(sub1)
        child2.append(sub2)

    children.append(child1)
    children.append(child2)

    return children


def make_mutation(parents, labels, probability=0.1):
    """
    Make the mutation between the parents

    :param parents: parents to mutate
    :param labels: labels of the linguistic variables
    :param probability: probability of mutation
    :return: children mutated
    """

    # choose if the mutation will be made
    if random.random() > probability:
        return parents

    # declare the labels

    # extract the parents
    parent1, parent2 = parents

    # # Perform crossover for each pair of parents
    children = []
    child1, child2 = [], []

    # mutate the individual
    child1 = mutate_individual(parent1, labels)
    child2 = mutate_individual(parent2, labels)

    # append the children
    children.append(child1)
    children.append(child2)

    return children


def mutate_individual(individual, labels):
    """
    Mutate the individual with simple change

    :param individual: individual to mutate
    :param labels: labelsfor swap
    :return: mutated individual
    """
    mutated_individual = []

    mutation_rate = random.random()

    for rule in individual:
        mutated_rule = []
        for value in rule:
            if random.random() < mutation_rate:
                mutated_value = random.choice(labels)
            else:
                mutated_value = value
            mutated_rule.append(mutated_value)
        mutated_individual.append(mutated_rule)

    return mutated_individual


def replace_worsts(population, replacement, indexes):
    """
    Replace the worsts individuals with the new ones

    :param population: population of the genetic algorithm
    :param fitness: fitness of the population
    :param replacement: new individuals to replace the worsts
    :return: new population
    """

    # replace the worsts
    for i in range(len(indexes)):
        population[indexes[i]] = replacement[i]

    return population
