import os
import ast
import csv
import math
import copy
import glob
import random
import operator
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from networkx.generators.random_graphs import watts_strogatz_graph, fast_gnp_random_graph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter


def sample_age():
    """Return an age group sampled from a distribution that is representative of the US"""
    p = random.random()
    if p < .121:
        return '0-9'
    elif p < .249:
        return '10-19'
    elif p < .386:
        return '20-29'
    elif p < .521:
        return '30-39'
    elif p < .644:
        return '40-49'
    elif p < .773:
        return '50-59'
    elif p < .888:
        return '60-69'
    elif p < .961:
        return '70-79'
    else:
        return '80-120'


def occupation(age):
    """Return an occupation that depends on the age group"""
    lower = int(age.split('-')[0])
    upper = int(age.split('-')[1])
    assert lower <= upper, 'The second value must be no smaller than the first one'
    if lower >= 0 and upper <= 19:
        return 'student'
    elif lower >= 20 and upper <= 69:
        return 'worker'
    elif lower >= 70 and upper <= 120:
        return 'retiree'
    else:
        raise ValueError('Not in the specified age groups')


class Node:
    """Individual with state information (health, activeness, output, etc.)"""

    personal_output = 1.0

    def __init__(self, age, personal_output_scale=1.0):
        assert personal_output_scale >= 0, 'Scale of personal output must be nonnegative'
        self.age = age
        self.occupation = occupation(self.age)
        self.connections = []  # Indices of everyone that the node interacts with outside home
        self.colleagues = []  # Indices of the node's colleagues
        self.household = None  # Household index
        self.vulnerable = False  # True if vulnerable to severe illness
        self.stay_at_home = False  # True if staying at home as required by the policy
        self.isolated = False  # True if isolated at home (symptomatic but not hospitalized)
        self.infected = False  # True if currently infected
        self.recovered = False  # True if already recovered
        self.hospitalized = False  # True if currently hospitalized
        self.undertreated = False  # True if ever undertreated
        self.ICU = False  # True if currently in the ICU
        self.deceased = False  # True if already deceased
        self.disease_length = 0  # Total number of days of infection before recovery if not deceased
        self.days_infected = 0  # Counter of days that the node has been infected if not deceased
        self.symptom_date = 0  # Date since infection on which the node becomes symptomatic
        self.hospital_date = 0  # Date since infection on which the node is hospitalized
        self.ICU_date = 0  # Date since infection on which the node is admitted to the ICU
        self.death_date = 0  # Date since infection on which the node deceases
        # Financial information: Everyone active initially
        self.active = True
        self.output = self.personal_output * personal_output_scale
        self.old_active = self.active
        self.old_output = self.output

    def infected_update_day(self):
        """Update health status of the infected node
        Return the updated status of the node.

        The function does *not* check if the node is currently infected and alive.
        Instead, this is done in Simulation.simulation_step.

        The updated health status will occur at the end of the day.
        In other words, the patient can still infect other people
        before the end of the day.
        """
        self.days_infected += 1
        if self.symptom_date == 0:  # Asymptomatic
            if self.days_infected >= self.disease_length:
                self.recovered = True
                self.infected = False
                self.isolated = False
                return 'recovered'
            return 'asymptomatic'
        elif self.days_infected < self.symptom_date:  # Pre-symptomatic
            return 'pre-symptomatic'
        else:  # Symptomatic
            if self.days_infected == self.symptom_date:
                self.isolated = True
            if self.hospital_date > 0:  # To be hospitalized if not yet
                if not self.hospitalized and self.days_infected >= self.hospital_date:  # Hospitalized
                    self.stay_at_home = False
                    self.isolated = False
                    self.hospitalized = True
                    self.active = False
                    self.output = 0
                if self.ICU_date > 0:  # To be admitted to the ICU if not yet
                    if not self.ICU and self.days_infected >= self.ICU_date:  # Admitted to the ICU
                        self.ICU = True
                    if self.death_date > 0:  # To decease if not yet
                        if not self.deceased and self.days_infected >= self.death_date:
                            if self.undertreated:
                                self.deceased = 'undertreatment'
                            else:
                                self.deceased = 'virus'
                            self.stay_at_home = False
                            self.isolated = False
                            self.infected = False
                            self.recovered = False
                            self.hospitalized = False
                            self.ICU = False
                            self.active = False
                            self.output = 0
                            self.old_active = False
                            self.old_output = 0
                            return self.deceased
                        elif self.ICU:
                            return 'ICU'
                        elif self.hospitalized:
                            return 'hospitalized'
                        return 'symptomatic'
                    elif self.days_infected >= self.disease_length:  # Recovers after ICU
                        self.recovered = True
                        self.infected = False
                        self.hospitalized = False
                        self.ICU = False
                        self.active = self.old_active
                        self.output = self.old_output
                        return 'recovered'
                    elif self.ICU:
                        return 'ICU'
                    elif self.hospitalized:
                        return 'hospitalized'
                    return 'symptomatic'
                elif self.days_infected >= self.disease_length:  # Recovers after hospitalization but without ICU
                    self.recovered = True
                    self.infected = False
                    self.hospitalized = False
                    self.active = self.old_active
                    self.output = self.old_output
                    return 'recovered'
                elif self.hospitalized:
                    return 'hospitalized'
                return 'symptomatic'
            elif self.days_infected >= self.disease_length:  # Recovers without hospitalization
                self.recovered = True
                self.infected = False
                self.isolated = False
                self.stay_at_home = False
                return 'recovered'
            return 'symptomatic'


class Simulation:
    """Simulate an epidemic over a network and its impact on the economy"""

    # The scale of remaining output when half of the workers stay at home
    output_scale_half_workers = .85
    # Fraction of total output generated by the rich
    total_output_rich_fraction = .45
    # Epidemiological parameters
    min_disease_length = 7
    max_disease_length = 28
    asymptomatic_rate = .35
    min_asymptomatic_length = 2
    max_asymptomatic_length = 10
    min_hospital_after_symptoms = 1
    max_hospital_after_symptoms = 12
    max_ICU_after_hospital = 14
    max_death_after_ICU = 14
    # Probability of hospitalization conditional on symptomatic infection
    hospitalization_rates = {'0-9': .001, '10-19': .003, '20-29': .012, '30-39': .032, '40-49': .049,
                             '50-59': .102, '60-69': .166, '70-79': .243, '80-120': .273}
    # Probability of ICU admission conditional on hospitalization
    ICU_rates = {'0-9': .05, '10-19': .05, '20-29': .05, '30-39': .05, '40-49': .063,
                 '50-59': .122, '60-69': .274, '70-79': .432, '80-120': .709}
    # Probability of death conditional on admission to the ICU
    death_rates = {'0-9': .615, '10-19': .615, '20-29': .769, '30-39': .769, '40-49': .748,
                   '50-59': .742, '60-69': .744, '70-79': .747, '80-120': .739}

    def __init__(self):
        self.time = 0
        self.time_steps_recorded = []
        self.household_grouping = False
        self.household_dict = {}
        self.max_household_sizes = {}
        self.num_nodes = 0
        self.num_nodes_poor = 0
        self.num_nodes_rich = 0
        self.vulnerable_population_fraction = 0
        self.vulnerability_indices = {'vulnerable': 1.0, 'normal': 1.0}
        self.node_dict = None
        self.hospitalized_nodes = []
        self.test_backlog = None
        self.initial_total_output = None
        self.rich_to_poor_output_ratio = 1.0  # Average ratio of a rich person's output to that of the poor
        self.personal_to_linkage_output_ratio = 1.0  # Ratio of personal output to average output from colleague linkage
        self.colleague_linkage_output = 1.0  # Output per colleague linkage
        self.inflection_x = 1.0  # x coordinate of the inflection point of the sigmoid function
        self.active_idx_to_node_idx = {}
        # Aggregate statistics
        self.stay_at_home_count = None  # Time series of the number of people staying at home
        self.isolation_count = None  # Time series of the number of people isolating at home
        self.infections = None  # Time series of the number of people that are currently infected
        self.recoveries = None  # Time series of the number of people that have already recovered
        self.hospitalizations = None  # Time series of the number of people that are currently hospitalized
        self.hospitalizations_cumsum = None  # Time series of the number of past, current, and future hospitalizations
        self.ICU_count = None  # Time series of the number of people that are currently in the ICU
        self.viral_deaths = None
        self.undertreated_deaths = None
        self.deaths_of_despair = None
        self.active_count = None
        self.total_output = None
        self.total_subsidy = None

    def init_data_types(self):
        """Initialize several attributes to the correct data type, depending on whether household grouping is used"""
        time_series = ['test_backlog', 'stay_at_home_count', 'isolation_count', 'infections', 'recoveries',
                       'hospitalizations', 'hospitalizations_cumsum', 'ICU_count',
                       'viral_deaths', 'undertreated_deaths', 'deaths_of_despair',
                       'active_count', 'total_output', 'total_subsidy']
        if not self.household_grouping:
            self.initial_total_output = 0
            for x in time_series:
                setattr(self, x, [])
        else:
            self.initial_total_output = {'poor': 0, 'rich': 0}
            for x in time_series:
                setattr(self, x, {'poor': [], 'rich': []})

    def populate_connections(self):
        """Combine various types of non-household connections of each node"""
        for i in range(self.num_nodes):
            self.node_dict[i].connections = copy.deepcopy(self.node_dict[i].colleagues)

    def household_network(self, num_households, max_household_size=5, household_type=None, personal_output_scale=1.0,
                          same_age=False):
        """Generate a network that comprises disconnected cliques which represent household connections
        If same_age is True, then members in the same household are in the same age group.

        Always the first network to generate if used.
        """
        assert num_households > 0 and num_households == int(num_households), 'Number of households must be ' \
                                                                             'a positive integer'
        assert max_household_size > 0 and max_household_size == int(max_household_size), 'Maximum household size must' \
                                                                                         ' be a positive integer'
        assert household_type in {None, 'poor', 'rich'}, 'Household type must be None, "poor", or "rich"'
        if household_type is not None:
            self.household_grouping = True
        self.init_data_types()
        self.max_household_sizes[household_type] = max_household_size
        household_sizes = np.random.randint(low=1, high=max_household_size + 1, size=num_households)
        num_nodes = int(np.sum(household_sizes))
        if same_age:
            household_ages = [sample_age() for _ in range(num_households)]
            ages = np.repeat(household_ages, household_sizes)
            if self.node_dict is None:
                self.node_dict = {i: Node(ages[i], personal_output_scale) for i in range(num_nodes)}
            else:
                self.node_dict.update({i: Node(ages[i - self.num_nodes], personal_output_scale)
                                       for i in range(self.num_nodes, self.num_nodes + num_nodes)})
        elif self.node_dict is None:
            self.node_dict = {i: Node(sample_age(), personal_output_scale) for i in range(num_nodes)}
        else:
            self.node_dict.update({i: Node(sample_age(), personal_output_scale)
                                   for i in range(self.num_nodes, self.num_nodes + num_nodes)})
        node_idx = self.num_nodes
        num_existing_households = len(self.household_dict)
        for h in range(num_households):
            household_members = set(range(node_idx, node_idx + household_sizes[h]))
            self.household_dict[h + num_existing_households] = {'type': household_type, 'members': household_members}
            for _ in range(household_sizes[h]):
                self.node_dict[node_idx].household = h + num_existing_households
                node_idx += 1
        if household_type == 'poor':
            self.num_nodes_poor = num_nodes
        elif household_type == 'rich':
            self.num_nodes_rich = num_nodes
        self.num_nodes += num_nodes
        del household_sizes

    def econ_network(self, func, args):
        """Generate a network that comprises professional connections

        args: A list of variables for func. The first element of args
        should be the number of nodes if no network exists
        """
        if self.num_nodes == 0:  # No existing network
            assert args[0] is not None, 'No network exists. Please specify the number of nodes.'
            assert args[0] == int(args[0]) and args[0] > 0, 'Number of nodes must be a positive integer'
            self.num_nodes = args[0]
            self.node_dict = {i: Node(sample_age()) for i in range(self.num_nodes)}
            self.graph_function(func, args, edge_type='econ')
        else:
            self.graph_function(func, args, edge_type='econ')

    def add_dict_active_idx_to_node_idx(self):
        """Create a dictionary that maps the index of an (initially) active node in the economic network
        to that in the whole network
        """
        assert self.num_nodes is not None, 'No network exists. Please first create a network.'
        count = 0
        for i in range(self.num_nodes):
            if self.node_dict[i].active:
                self.active_idx_to_node_idx[count] = i
                count += 1

    def graph_function(self, func, args, edge_type=None):
        """Wrapper that can be used to create an undirected network

        func creates a network (see networkx documentation or functions above)
        args are the arguments of that particular function
        """
        assert edge_type in {None, 'econ'}, 'Edge type must be either None or "econ"'
        if self.num_nodes == 0:
            self.init_data_types()
        num_active = None
        if edge_type == 'econ':
            self.add_dict_active_idx_to_node_idx()
            num_active = len(self.active_idx_to_node_idx)
            if args[0] is None or isinstance(args[0], int):  # Ignore the number of nodes if a network already exists
                args[0] = num_active

        def connection_type(idx):
            if edge_type == 'econ':
                return self.node_dict[idx].colleagues
            elif edge_type is None:
                return self.node_dict[idx].connections

        res = func(*args)
        if isinstance(res, (list, np.ndarray, np.matrix)):
            a = np.array(res)
            assert a.ndim == 2 and a.shape[0] == a.shape[1], 'Output of func must be a square matrix'
            assert np.allclose(a, a.T), 'Adjacency matrix must be symmetric'
            assert ((a == 0) | (a == 1)).all(), 'Adjacency matrix must be binary'
            if edge_type == 'econ':
                for i in range(num_active - 1):
                    for j in range(i + 1, num_active):
                        if a[i, j] == 1:
                            i_idx = self.active_idx_to_node_idx[i]
                            j_idx = self.active_idx_to_node_idx[j]
                            connection_type(i_idx).append(j_idx)
                            connection_type(j_idx).append(i_idx)
            else:
                for i in range(self.num_nodes - 1):
                    for j in range(i + 1, self.num_nodes):
                        if a[i, j] == 1:
                            connection_type(i).append(j)
                            connection_type(j).append(i)
            del res, a
        elif isinstance(res, nx.Graph):
            assert not nx.is_directed(res), 'Output of func must be an undirected network'
            if edge_type == 'econ':
                for e in res.edges:
                    if e[0] != e[1]:
                        i_idx = self.active_idx_to_node_idx[e[0]]
                        j_idx = self.active_idx_to_node_idx[e[1]]
                        connection_type(i_idx).append(j_idx)
                        connection_type(j_idx).append(i_idx)
            else:
                for e in res.edges:
                    if e[0] != e[1]:
                        connection_type(e[0]).append(e[1])
                        connection_type(e[1]).append(e[0])
            del res
        else:
            raise TypeError('Output of func must be a matrix or a network')

    def add_vulnerable_group(self, population_fraction=.01, vulnerability=.05):
        """Make a fraction of the population vulnerable to severe illness"""
        if not self.household_grouping:
            assert 0 < population_fraction < 1, 'Fraction of vulnerable population must be between 0 and 1 exclusive'
        else:
            population_poor_fraction = self.num_nodes_poor / self.num_nodes
            assert 0 < population_fraction <= population_poor_fraction, 'Fraction of vulnerable population must be ' \
                                                                        'positive and no more than the fraction of ' \
                                                                        'the poor'
        assert vulnerability > 0, 'Vulnerability must be positive'
        max_hospitalization_rate = max(self.hospitalization_rates.values())
        max_ICU_rate = max(self.ICU_rates.values())
        max_death_rate = max(self.death_rates.values())
        max_param = max(1 - self.asymptomatic_rate, max_hospitalization_rate, max_ICU_rate, max_death_rate)
        assert max_param * (1 + vulnerability) <= 1, 'Vulnerability need to be sufficiently low so that modified ' \
                                                     'rates are between 0 and 1'
        assert ((1 + vulnerability) * population_fraction) <= 1, 'Fraction of vulnerable population and ' \
                                                                 'vulnerability need to be sufficiently low so that ' \
                                                                 'modified rates are between 0 and 1'
        self.vulnerability_indices['vulnerable'] = 1 + vulnerability
        self.vulnerability_indices['normal'] = 1 - vulnerability * population_fraction / (1 - population_fraction)
        self.vulnerable_population_fraction = population_fraction
        p = np.random.rand(self.num_nodes)
        if not self.household_grouping:
            for i in range(self.num_nodes):
                if p[i] < population_fraction:
                    self.node_dict[i].vulnerable = True
        else:
            poor_fraction = population_fraction * self.num_nodes / self.num_nodes_poor
            for i in range(self.num_nodes):
                if p[i] < poor_fraction:
                    node = self.node_dict[i]
                    if self.household_dict[node.household]['type'] == 'poor':
                        node.vulnerable = True

    def calc_rich_to_poor_output_ratio(self):
        """Calibrate the average ratio of a rich person's output to that of the poor"""
        assert min(self.num_nodes_poor, self.num_nodes_rich) > 0, 'Network must include both poor and rich households'
        population_poor_fraction = self.num_nodes_poor / self.num_nodes
        self.rich_to_poor_output_ratio = (self.total_output_rich_fraction * population_poor_fraction
                                          / ((1 - self.total_output_rich_fraction) * (1 - population_poor_fraction)))
        for h in self.household_dict:
            if self.household_dict[h]['type'] == 'rich':
                for i in self.household_dict[h]['members']:
                    node = self.node_dict[i]
                    node.output *= self.rich_to_poor_output_ratio
                    node.old_output *= self.rich_to_poor_output_ratio

    def stay_at_home_by_occupation_policy(self, occupations, output_remaining_scale=1.0):
        """Let the specified occupation(s) stay at home"""
        assert 0 <= output_remaining_scale <= 1, 'The scale of remaining output must be between 0 and 1'
        for i in range(self.num_nodes):
            node = self.node_dict[i]
            if node.occupation in occupations:
                node.stay_at_home = True
                node.output *= output_remaining_scale

    def partial_opening_policy(self, p_stay_at_home, output_remaining_scale=1.0):
        """Let a fraction of workers stay at home"""
        assert 0 <= p_stay_at_home <= 1, 'The probability of staying at home must be between 0 and 1'
        assert 0 <= output_remaining_scale <= 1, 'The scale of remaining output must be between 0 and 1'
        p = np.random.rand(self.num_nodes)
        for i in range(self.num_nodes):
            node = self.node_dict[i]
            if node.occupation == 'worker' and p[i] < p_stay_at_home:
                node.stay_at_home = True
                node.output *= output_remaining_scale

    def reopen_policy(self):
        """Let all healthy and asymptomatic workers go back to work"""
        for i in range(self.num_nodes):
            node = self.node_dict[i]
            if node.occupation == 'worker' and not node.isolated and not node.hospitalized and not node.deceased:
                node.stay_at_home = False
                node.active = node.old_active
                node.output = node.old_output

    def infect_node(self, target_node_idx, p_trans):
        """Infect a susceptible target node probabilistically

        Return True if the target node is infected and False otherwise.

        p_trans: Probability of transmission of the disease to the target node
        """
        assert 0 <= p_trans <= 1, 'Probability of transmission must be between 0 and 1'
        node = self.node_dict[target_node_idx]
        if node.infected or node.deceased:
            return False
        elif node.recovered:  # Assume that recovered patients cannot be reinfected
            return False
        else:  # Susceptible
            if random.random() < p_trans:
                node.infected = True
                node.disease_length = random.randint(self.min_disease_length, self.max_disease_length)
                if self.vulnerable_population_fraction == 0:  # No vulnerable group
                    asymptomatic_rate = self.asymptomatic_rate
                    hospitalization_rate = self.hospitalization_rates[node.age]
                    ICU_rate = self.ICU_rates[node.age]
                    death_rate = self.death_rates[node.age]
                elif node.vulnerable:
                    asymptomatic_rate = 1 - (1 - self.asymptomatic_rate) * self.vulnerability_indices['vulnerable']
                    hospitalization_rate = (self.hospitalization_rates[node.age]
                                            * self.vulnerability_indices['vulnerable'])
                    ICU_rate = self.ICU_rates[node.age] * self.vulnerability_indices['vulnerable']
                    death_rate = self.death_rates[node.age] * self.vulnerability_indices['vulnerable']
                else:
                    asymptomatic_rate = 1 - (1 - self.asymptomatic_rate) * self.vulnerability_indices['normal']
                    hospitalization_rate = self.hospitalization_rates[node.age] * self.vulnerability_indices['normal']
                    ICU_rate = self.ICU_rates[node.age] * self.vulnerability_indices['normal']
                    death_rate = self.death_rates[node.age] * self.vulnerability_indices['normal']
                if random.random() >= asymptomatic_rate:
                    node.symptom_date = 1 + random.randint(self.min_asymptomatic_length, self.max_asymptomatic_length)
                    if random.random() < hospitalization_rate:
                        node.hospital_date = node.symptom_date + random.randint(self.min_hospital_after_symptoms,
                                                                                self.max_hospital_after_symptoms)
                        if random.random() < ICU_rate:
                            node.ICU_date = node.hospital_date + random.randint(0, self.max_ICU_after_hospital)
                            if random.random() < death_rate:
                                node.death_date = node.ICU_date + random.randint(0, self.max_death_after_ICU)
                                node.disease_length += node.death_date
                            else:
                                node.disease_length += node.ICU_date
                        else:
                            node.disease_length += node.hospital_date
                    else:
                        node.disease_length += node.symptom_date
                status = node.infected_update_day()
                if status in {'hospitalized', 'ICU'}:
                    self.hospitalized_nodes.append(node)
                return True
            return False

    def check_hospital_capacity(self, hospital_capacity=2.5e-3, undertreatment_effect=.5):
        """Check if the number of hospitalizations exceeds the capacity

        If the capacity is exceeded, then hospitalized people will be at increased risk for severe illness.
        Specifically, the rates of ICU admission (for patients not yet in ICU) and fatality will increase as specified
        by the undertreatment effect. If household grouping is used, then only the poor are affected by undertreatment.
        """
        assert 0 < hospital_capacity <= 1, 'Capacity must be positive and no more than 1'
        assert undertreatment_effect >= 0, 'The effect of undertreatment must be nonnegative'
        over_capacity = len(self.hospitalized_nodes) / (self.num_nodes * hospital_capacity) - 1
        if over_capacity > 0:
            factor = 1 + undertreatment_effect * over_capacity
            undertreated_nodes = []
            for node in self.hospitalized_nodes:
                if not node.undertreated and self.household_dict[node.household]['type'] != 'rich':
                    undertreated_nodes.append(node)
            if undertreated_nodes:
                for node in undertreated_nodes:
                    node.undertreated = True
                    if self.vulnerable_population_fraction == 0:  # No vulnerable group
                        ICU_rate = self.ICU_rates[node.age] * factor
                        death_rate = self.death_rates[node.age] * factor
                    elif node.vulnerable:
                        ICU_rate = self.ICU_rates[node.age] * self.vulnerability_indices['vulnerable'] * factor
                        death_rate = self.death_rates[node.age] * self.vulnerability_indices['vulnerable'] * factor
                    else:
                        ICU_rate = self.ICU_rates[node.age] * self.vulnerability_indices['normal'] * factor
                        death_rate = self.death_rates[node.age] * self.vulnerability_indices['normal'] * factor
                    if node.ICU:
                        if random.random() < death_rate:
                            node.death_date = node.ICU_date + random.randint(0, self.max_death_after_ICU)
                    else:
                        if random.random() < ICU_rate:
                            node.ICU_date = node.hospital_date + random.randint(0, self.max_ICU_after_hospital)
                            if random.random() < death_rate:
                                node.death_date = node.ICU_date + random.randint(0, self.max_death_after_ICU)
            del undertreated_nodes

    def calc_output_measures(self, average_worker_degree):
        """Calibrate measures that are related to output and deaths of despair"""
        assert .25 < self.output_scale_half_workers < 1, 'output_scale_half_workers must be between 0.25 and 1 ' \
                                                         'exclusive'
        assert average_worker_degree > 0, 'Average worker degree must be positive'
        self.personal_to_linkage_output_ratio = ((self.output_scale_half_workers - .25)
                                                 / (1 - self.output_scale_half_workers))
        self.colleague_linkage_output = Node.personal_output / (self.personal_to_linkage_output_ratio
                                                                * average_worker_degree)
        self.inflection_x = Node.personal_output / (2 * self.personal_to_linkage_output_ratio)

    def sigmoid(self, x, nu=1e-3):
        """Sigmoid function for probabilistically generating deaths of despair"""
        assert nu > 0, 'Parameter nu must be positive'
        res = (1 + nu ** (x / self.inflection_x)) ** (-1 / nu)
        return res

    def check_despair(self, p_despair_factor=5.5e-5, subsidy=None):
        """Check output per capita in each household, subsidize nodes in need, and sample deaths of despair

        If subsidy is None, then subsidy will be given to each node as needed to reach the sufficient output.
        Otherwise, a uniform subsidy will be given to each node that has less than the sufficient output
        in the amount as specified.
        """
        assert 0 <= p_despair_factor <= 1, 'The factor of probability of despair must be between 0 and 1'
        if subsidy is not None:
            legal_types = {int, np.int, np.int8, np.int16, np.int32, np.int64,
                           float, np.float, np.float16, np.float32, np.float64, np.float128}
            assert type(subsidy) in legal_types and subsidy >= 0, 'Subsidy must be None or a nonnegative number'
        sufficient_output = (1 + 1 / self.personal_to_linkage_output_ratio) * Node.personal_output
        if not self.household_grouping:
            total_subsidy = 0
        else:
            total_subsidy = {'poor': 0, 'rich': 0}
        for h, household in self.household_dict.items():
            household_members = household['members']
            household_type = household['type']
            # Check if the household currently faces despair
            num_at_home = len(household_members)
            household_output = 0
            for i in household_members:
                node = self.node_dict[i]
                output = 0
                if node.hospitalized or node.deceased:
                    num_at_home -= 1
                else:
                    if node.active:
                        output += node.output
                        if not node.stay_at_home and not node.isolated:
                            for j in node.colleagues:
                                colleague = self.node_dict[j]
                                if colleague.active and not colleague.stay_at_home and not colleague.isolated:
                                    if household_type == 'rich':
                                        output += self.colleague_linkage_output * self.rich_to_poor_output_ratio
                                    else:
                                        output += self.colleague_linkage_output
                    if output < sufficient_output:
                        if subsidy is None:  # Subsidize each node as needed
                            if not self.household_grouping:
                                total_subsidy += sufficient_output - output
                            else:
                                total_subsidy[household_type] += sufficient_output - output
                            household_output += sufficient_output
                        else:
                            if not self.household_grouping:
                                total_subsidy += subsidy
                            else:
                                total_subsidy[household_type] += subsidy
                            household_output += output + subsidy
                    else:
                        household_output += output
            if num_at_home > 0:
                output_loss = sufficient_output - household_output / num_at_home
                if output_loss > 0:
                    p_despair = self.sigmoid(output_loss) * p_despair_factor
                    for i in household_members:
                        node = self.node_dict[i]
                        if node.hospitalized or node.deceased:
                            continue
                        if random.random() < p_despair:
                            node.deceased = 'despair'
                            node.active = False
                            node.output = 0
                            node.old_active = False
                            node.old_output = 0
        return total_subsidy

    def check_despair_limited_subsidy(self, p_despair_factor=5.5e-5, subsidy=.3,
                                      subsidy_population_factor=.1, allocation='greedy'):
        """Check output per capita in each household, subsidize nodes in need subject to a total budget,
        and sample deaths of despair

        If allocation is greedy, then a fixed subsidy will be given to each node among the ones that have the low output.
        Otherwise, a uniform subsidy will be given to each node that has less than the sufficient output
        in the amount as specified.
        """
        assert 0 <= p_despair_factor <= 1, 'The factor of probability of despair must be between 0 and 1'
        legal_types = {int, np.int, np.int8, np.int16, np.int32, np.int64,
                       float, np.float, np.float16, np.float32, np.float64, np.float128}
        assert type(subsidy) in legal_types and subsidy >= 0, 'Subsidy must be a nonnegative number'
        assert type(subsidy_population_factor) in legal_types and 0 <= subsidy_population_factor <= 1, \
            'subsidy_population_factor must be between 0 and 1'
        assert allocation in {'greedy', 'random'}, 'Allocation can be either greedy or random'
        num_subsidies = int(subsidy_population_factor * self.num_nodes)  # Number of subsidies available per day
        sufficient_output = (1 + 1 / self.personal_to_linkage_output_ratio) * Node.personal_output

        nodes_needing_subsidy = {}  # Key: node ID. Value: the amount of subsidy needed to reach sufficient output
        household_finance = {}
        if not self.household_grouping:
            total_subsidy = 0
        else:
            total_subsidy = {'poor': 0, 'rich': 0}
        for h, household in self.household_dict.items():
            household_members = household['members']
            household_type = household['type']
            # Check if the household currently faces despair
            num_at_home = len(household_members)
            household_output = 0
            for i in household_members:
                node = self.node_dict[i]
                output = 0
                if node.hospitalized or node.deceased:
                    num_at_home -= 1
                else:
                    if node.active:
                        output += node.output
                        if not node.stay_at_home and not node.isolated:
                            for j in node.colleagues:
                                colleague = self.node_dict[j]
                                if colleague.active and not colleague.stay_at_home and not colleague.isolated:
                                    if household_type == 'rich':
                                        output += self.colleague_linkage_output * self.rich_to_poor_output_ratio
                                    else:
                                        output += self.colleague_linkage_output
                        household_output += output
                    if output < sufficient_output:
                        nodes_needing_subsidy[i] = sufficient_output - output
            if num_at_home > 0:
                household_finance[h] = {'num_at_home': num_at_home, 'household_output': household_output}
        # Decide on nodes to subsidize
        if len(nodes_needing_subsidy) <= num_subsidies:
            nodes_to_subsidize = list(nodes_needing_subsidy.keys())
        else:
            if allocation == 'greedy':
                sorted_dict = sorted(nodes_needing_subsidy.items(), key=operator.itemgetter(1), reverse=True)
                nodes_to_subsidize = [sorted_dict[i][0] for i in range(num_subsidies)]
            else:  # Random allocation
                nodes_to_subsidize = np.random.choice(list(nodes_needing_subsidy.keys()), num_subsidies, replace=False)
        # Subsidize chosen nodes and calculate total subsidies
        if not self.household_grouping:
            total_subsidy = subsidy * len(nodes_to_subsidize)
            for i in nodes_to_subsidize:
                node = self.node_dict[i]
                household_finance[node.household]['household_output'] += subsidy
        else:
            for i in nodes_to_subsidize:
                node = self.node_dict[i]
                household_finance[node.household]['household_output'] += subsidy
                total_subsidy[self.household_dict[node.household]['type']] += subsidy
        # Sample deaths of despair
        for h, record in household_finance.items():
            output_loss = sufficient_output - record['household_output'] / record['num_at_home']
            if output_loss > 0:
                p_despair = self.sigmoid(output_loss) * p_despair_factor
                for i in self.household_dict[h]['members']:
                    node = self.node_dict[i]
                    if node.hospitalized or node.deceased:
                        continue
                    if random.random() < p_despair:
                        node.deceased = 'despair'
                        node.active = False
                        node.output = 0
                        node.old_active = False
                        node.old_output = 0

        del nodes_needing_subsidy
        del nodes_to_subsidize
        del household_finance

        return total_subsidy

    def check_output(self, total_subsidy, p_inactive_factor=1e-2):
        """Probabilistically lay off workers if the total subsidy does not cover the total output loss"""
        if type(total_subsidy) in {float, int}:
            assert total_subsidy >= 0, 'Total subsidy must be nonnegative'
        else:
            assert isinstance(total_subsidy, dict), 'Total subsidy must be either a nonnegative number or a dictionary'
            assert min(total_subsidy.values()) >= 0, 'Total subsidy must be nonnegative for every household group'
        assert 0 <= p_inactive_factor <= 1, 'The factor of probability of becoming inactive must be between 0 and 1'
        if not self.household_grouping:
            initial_total_output = self.initial_total_output
        else:
            initial_total_output = self.initial_total_output['poor'] + self.initial_total_output['rich']
        assert initial_total_output > 0, 'Initial total output must be positive'
        if not self.household_grouping:
            normalized_loss = 1 - (total_subsidy + self.get_total_output()) / initial_total_output
        else:
            total_output = self.get_total_output()
            normalized_loss = 1 - (total_subsidy['poor'] + total_subsidy['rich']
                                   + total_output['poor'] + total_output['rich']) / initial_total_output
        if normalized_loss > 0:
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                if node.occupation == 'worker':
                    if node.deceased or node.hospitalized:  # Assume that no one can get laid off while hospitalized
                        continue
                    if random.random() < p_inactive_factor * normalized_loss:
                        node.active = False
                        node.output = 0
                        node.old_active = False
                        node.old_output = 0

    def contact_tracing(self, target_node_idx, efficacy=.7):
        """Return people who may have been infected by the target node
        If household type is specified, then a dictionary is returned. Otherwise, a set is returned.
        """
        assert 0 <= efficacy <= 1, 'Efficacy must be between 0 and 1'
        node = self.node_dict[target_node_idx]
        household_members = copy.deepcopy(self.household_dict[node.household]['members'])
        household_members.remove(target_node_idx)
        household_type = self.household_dict[node.household]['type']
        if not self.household_grouping:
            contacts = set()
        else:
            contacts = {'poor': set(), 'rich': set()}
        for j in household_members:
            roommate = self.node_dict[j]
            if roommate.isolated or roommate.hospitalized or roommate.deceased:
                continue
            if household_type is None:
                contacts.add(j)
            elif household_type == 'poor':
                contacts['poor'].add(j)
            elif household_type == 'rich':
                contacts['rich'].add(j)
        if node.active and not node.stay_at_home:
            for j in node.connections:
                colleague = self.node_dict[j]
                if colleague.stay_at_home or colleague.isolated or colleague.hospitalized or colleague.deceased:
                    continue
                if colleague.active and random.random() < efficacy:
                    colleague_household_type = self.household_dict[colleague.household]['type']
                    if colleague_household_type is None:
                        contacts.add(j)
                    elif colleague_household_type == 'poor':
                        contacts['poor'].add(j)
                    elif colleague_household_type == 'rich':
                        contacts['rich'].add(j)
        return contacts

    def get_test_candidates(self):
        """Return people who are not known to have been infected.
        If household type is specified, then a dictionary is returned. Otherwise, a set is returned.
        """
        if not self.household_grouping:
            candidates = set()
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                if not node.isolated and not node.hospitalized and not node.deceased:
                    candidates.add(i)
        else:
            candidates = {'poor': set(), 'rich': set()}
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                if not node.isolated and not node.hospitalized and not node.deceased:
                    household_type = self.household_dict[node.household]['type']
                    candidates[household_type].add(i)
        return candidates

    # noinspection PyTypeChecker
    def viral_test(self, nodes_to_test, sensitivity=.9, contact_tracing_efficacy=.7):
        """Conduct viral tests on all nodes in the order provided
        Once an infected person is newly detected, the person will be isolated at home, and the person's contacts
        will be prioritized in testing.

        nodes_to_test: a list of nodes to be tested in the descending order of priority
        sensitivity: ability to detect a positive case, equal to one minus false-negative rate
        """
        assert isinstance(nodes_to_test, (list, tuple, np.ndarray)), 'Nodes to test must be a list, tuple, ' \
                                                                     'or numpy array'
        assert len(nodes_to_test) > 0, 'Nodes to test cannot be empty'
        assert 0 <= sensitivity <= 1, 'Sensitivity must be between 0 and 1'
        new_positives = set()
        p = np.random.rand(len(nodes_to_test))
        for idx, i in enumerate(nodes_to_test):
            node = self.node_dict[i]
            if p[idx] < sensitivity and node.infected:
                new_positives.add(i)
                node.isolated = True
                contacts = self.contact_tracing(i, contact_tracing_efficacy)
                if not self.household_grouping:
                    self.test_backlog.extend(j for j in contacts if j not in self.test_backlog)
                else:
                    self.test_backlog['poor'].extend(j for j in contacts['poor'] if j not in self.test_backlog['poor'])
                    self.test_backlog['rich'].extend(j for j in contacts['rich'] if j not in self.test_backlog['rich'])
        if not self.household_grouping:
            self.test_backlog = [j for j in self.test_backlog if j not in new_positives]
        else:
            self.test_backlog['poor'] = [j for j in self.test_backlog['poor'] if j not in new_positives]
            self.test_backlog['rich'] = [j for j in self.test_backlog['rich'] if j not in new_positives]
        del new_positives

    # noinspection PyTypeChecker
    def random_testing_protocol(self, num_tests, sensitivity=.9, contact_tracing_efficacy=.7):
        """Conduct random testing among eligible candidates, prioritizing those in the backlog.

        If num_tests is an integer, then testing is conducted on the whole population uniformly at random.
        If num_tests is a dictionary with the key being household type, then random testing will be conducted within
        each household type as specified.
        """
        if not type(num_tests) == int or not num_tests >= 0:
            assert type(num_tests) == dict, 'Number of tests must be either a nonnegative integer or a dictionary'
            assert self.household_grouping, 'Household grouping must be used for differential testing'
            for h in ['poor', 'rich']:
                assert h in num_tests, f'Number of tests must have {h} as a key'
                assert type(num_tests[h]) == int and num_tests[h] >= 0, 'Number of tests must be either a nonnegative' \
                                                                        ' integer or a dictionary'
        if type(num_tests) == int:
            total_num_tests = num_tests
        else:
            total_num_tests = num_tests['poor'] + num_tests['rich']
        if total_num_tests > 0:
            nodes_to_test = []
            candidates = self.get_test_candidates()
            if not self.household_grouping:
                num_candidates = len(candidates)
            else:
                num_candidates = len(candidates['poor']) + len(candidates['rich'])
            if num_candidates == 0:
                print(f'[Day {self.time}] Warning: No candidates left for testing.')
            elif num_candidates < total_num_tests:
                if not self.household_grouping:
                    nodes_to_test = list(candidates)
                else:
                    nodes_to_test = list(candidates['poor']) + list(candidates['rich'])
                print(f'[Day {self.time}] Warning: Fewer than {total_num_tests} candidates left. '
                      'Everyone testable is tested.')
            else:
                if not self.household_grouping:
                    self.test_backlog = [i for i in self.test_backlog if i in candidates]
                else:
                    self.test_backlog['poor'] = [i for i in self.test_backlog['poor'] if i in candidates['poor']]
                    self.test_backlog['rich'] = [i for i in self.test_backlog['rich'] if i in candidates['rich']]
                if type(num_tests) == int:
                    if not self.household_grouping:
                        test_backlog = self.test_backlog
                    else:
                        test_backlog = self.test_backlog['poor'] + self.test_backlog['rich']
                    if len(test_backlog) < num_tests:
                        if self.household_grouping:
                            candidates = list(candidates['poor']) + list(candidates['rich'])
                        candidates = [i for i in candidates if i not in test_backlog]
                        nodes_to_test = test_backlog + list(np.random.choice(candidates, num_tests - len(test_backlog),
                                                                             replace=False))
                    else:
                        nodes_to_test = test_backlog[:num_tests]
                else:
                    if len(candidates['poor']) >= num_tests['poor'] and len(candidates['rich']) >= num_tests['rich']:
                        if len(self.test_backlog['poor']) < num_tests['poor']:
                            candidates_poor = [i for i in candidates['poor'] if i not in self.test_backlog['poor']]
                            num_samples = num_tests['poor'] - len(self.test_backlog['poor'])
                            sampled_candidates = list(np.random.choice(candidates_poor, num_samples, replace=False))
                            nodes_to_test_poor = self.test_backlog['poor'] + sampled_candidates
                        else:
                            nodes_to_test_poor = self.test_backlog['poor'][:num_tests['poor']]
                        if len(self.test_backlog['rich']) < num_tests['rich']:
                            candidates_rich = [i for i in candidates['rich'] if i not in self.test_backlog['rich']]
                            num_samples = num_tests['rich'] - len(self.test_backlog['rich'])
                            sampled_candidates = list(np.random.choice(candidates_rich, num_samples, replace=False))
                            nodes_to_test_rich = self.test_backlog['rich'] + sampled_candidates
                        else:
                            nodes_to_test_rich = self.test_backlog['rich'][:num_tests['rich']]
                    elif len(candidates['poor']) < num_tests['poor']:
                        nodes_to_test_poor = list(candidates['poor'])
                        num_tests_rich = num_tests['rich'] + num_tests['poor'] - len(candidates['poor'])
                        if len(self.test_backlog['rich']) < num_tests_rich:
                            candidates_rich = [i for i in candidates['rich'] if i not in self.test_backlog['rich']]
                            num_samples = num_tests_rich - len(self.test_backlog['rich'])
                            sampled_candidates = list(np.random.choice(candidates_rich, num_samples, replace=False))
                            nodes_to_test_rich = self.test_backlog['rich'] + sampled_candidates
                        else:
                            nodes_to_test_rich = self.test_backlog['rich'][:num_tests_rich]
                    else:
                        nodes_to_test_rich = list(candidates['rich'])
                        num_tests_poor = num_tests['poor'] + num_tests['rich'] - len(candidates['rich'])
                        if len(self.test_backlog['poor']) < num_tests_poor:
                            candidates_poor = [i for i in candidates['poor'] if i not in self.test_backlog['poor']]
                            num_samples = num_tests_poor - len(self.test_backlog['poor'])
                            sampled_candidates = list(np.random.choice(candidates_poor, num_samples, replace=False))
                            nodes_to_test_poor = self.test_backlog['poor'] + sampled_candidates
                        else:
                            nodes_to_test_poor = self.test_backlog['poor'][:num_tests_poor]
                    nodes_to_test = np.concatenate((nodes_to_test_poor, nodes_to_test_rich))
            if len(nodes_to_test) > 0:
                if not self.household_grouping:
                    self.test_backlog = [i for i in self.test_backlog if i not in nodes_to_test]
                else:
                    self.test_backlog['poor'] = [i for i in self.test_backlog['poor'] if i not in nodes_to_test]
                    self.test_backlog['rich'] = [i for i in self.test_backlog['rich'] if i not in nodes_to_test]
                self.viral_test(nodes_to_test, sensitivity, contact_tracing_efficacy)

    def seed_simulation(self, initial_infections, record_stats=True):
        """Set initial infections and record initial values of various statistics if desired"""
        assert initial_infections is not None and len(initial_infections) > 0, 'Initial infections cannot be empty'
        assert len(initial_infections) == len(set(initial_infections)), 'Initial infections must be unique nodes'
        for i in initial_infections:
            self.infect_node(i, 1)
        self.initial_total_output = copy.deepcopy(self.get_total_output())
        if record_stats:
            if not self.household_grouping:
                self.record_aggregate_stats(0)
            else:
                self.record_aggregate_stats({'poor': 0, 'rich': 0})

    def simulation_step(self, p_trans_household, p_trans_other, num_viral_tests, contact_tracing_efficacy=.7,
                        hospital_capacity=2.5e-3, undertreatment_effect=.5,
                        p_despair_factor=5.5e-5, subsidy=None, subsidy_population_factor=.1, allocation=None,
                        p_inactive_factor=1e-2, record_stats=True):
        """Execute one time step of simulation"""
        self.time += 1
        self.hospitalized_nodes = []
        newly_infected = []
        for i in range(self.num_nodes):
            node = self.node_dict[i]
            if i in newly_infected:  # Assume that no one can get infected and infect others on the same day
                continue
            if node.deceased or node.recovered:
                continue
            if node.hospitalized:  # Assume that patients cannot infect others once hospitalized
                status = node.infected_update_day()
                if status in {'hospitalized', 'ICU'}:
                    self.hospitalized_nodes.append(node)
                continue
            if node.infected:
                household_members = self.household_dict[node.household]['members']
                for j in household_members:
                    infected = self.infect_node(j, p_trans_household)
                    if infected:
                        newly_infected.append(j)
                if node.active and not node.stay_at_home and not node.isolated:
                    for j in node.connections:
                        if not self.node_dict[j].active or self.node_dict[j].stay_at_home or self.node_dict[j].isolated:
                            continue
                        infected = self.infect_node(j, p_trans_other)
                        if infected:
                            newly_infected.append(j)
                status = node.infected_update_day()
                if status in {'hospitalized', 'ICU'}:
                    self.hospitalized_nodes.append(node)
        del newly_infected
        self.random_testing_protocol(num_viral_tests, contact_tracing_efficacy=contact_tracing_efficacy)
        self.check_hospital_capacity(hospital_capacity, undertreatment_effect)
        if allocation is None:
            total_subsidy = self.check_despair(p_despair_factor, subsidy)
        else:
            total_subsidy = self.check_despair_limited_subsidy(p_despair_factor, subsidy,
                                                               subsidy_population_factor, allocation)
        self.check_output(total_subsidy, p_inactive_factor)
        if record_stats:
            self.record_aggregate_stats(total_subsidy)

    def get_total_output(self):
        """Return the current total output if there is no household grouping and by group otherwise"""
        if not self.household_grouping:
            res = 0
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                if node.active:
                    res += node.output
                    if not node.stay_at_home and not node.isolated:
                        for j in node.colleagues:
                            colleague = self.node_dict[j]
                            if colleague.active and not colleague.stay_at_home and not colleague.isolated:
                                res += self.colleague_linkage_output
        else:
            res = {'poor': 0, 'rich': 0}
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                household_type = self.household_dict[node.household]['type']
                if node.active:
                    res[household_type] += node.output
                    if not node.stay_at_home and not node.isolated:
                        for j in node.colleagues:
                            colleague = self.node_dict[j]
                            if colleague.active and not colleague.stay_at_home and not colleague.isolated:
                                if self.household_dict[node.household]['type'] == 'rich':
                                    res[household_type] += (self.colleague_linkage_output
                                                            * self.rich_to_poor_output_ratio)
                                else:
                                    res[household_type] += self.colleague_linkage_output
        return res

    def record_aggregate_stats(self, total_subsidy):
        """Record aggregate statistics of simulation"""
        if not self.household_grouping:
            num_stay_at_home = 0
            num_isolated = 0
            num_infected = 0
            num_recovered = 0
            num_hospitalized_cumsum = 0
            num_ICU = 0
            num_viral_deaths = 0
            num_undertreated_deaths = 0
            num_deaths_of_despair = 0
            num_active = 0
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                if node.stay_at_home:
                    num_stay_at_home += 1
                if node.isolated:
                    num_isolated += 1
                if node.infected:
                    num_infected += 1
                if node.recovered:
                    num_recovered += 1
                if node.hospital_date > 0:
                    num_hospitalized_cumsum += 1
                if node.ICU:
                    num_ICU += 1
                if node.deceased == 'virus':
                    num_viral_deaths += 1
                if node.deceased == 'undertreatment':
                    num_undertreated_deaths += 1
                if node.deceased == 'despair':
                    num_deaths_of_despair += 1
                if node.active:
                    num_active += 1
            self.time_steps_recorded.append(self.time)
            self.stay_at_home_count.append(num_stay_at_home)
            self.isolation_count.append(num_isolated)
            self.infections.append(num_infected)
            self.recoveries.append(num_recovered)
            self.hospitalizations.append(len(self.hospitalized_nodes))
            self.hospitalizations_cumsum.append(num_hospitalized_cumsum)
            self.ICU_count.append(num_ICU)
            self.viral_deaths.append(num_viral_deaths)
            self.undertreated_deaths.append(num_undertreated_deaths)
            self.deaths_of_despair.append(num_deaths_of_despair)
            self.active_count.append(num_active)
            self.total_output.append(self.get_total_output())
            self.total_subsidy.append(total_subsidy)
        else:
            num_stay_at_home = {'poor': 0, 'rich': 0}
            num_isolated = {'poor': 0, 'rich': 0}
            num_infected = {'poor': 0, 'rich': 0}
            num_recovered = {'poor': 0, 'rich': 0}
            num_hospitalized = {'poor': 0, 'rich': 0}
            num_hospitalized_cumsum = {'poor': 0, 'rich': 0}
            num_ICU = {'poor': 0, 'rich': 0}
            num_viral_deaths = {'poor': 0, 'rich': 0}
            num_undertreated_deaths = {'poor': 0, 'rich': 0}
            num_deaths_of_despair = {'poor': 0, 'rich': 0}
            num_active = {'poor': 0, 'rich': 0}
            for i in range(self.num_nodes):
                node = self.node_dict[i]
                household_type = self.household_dict[node.household]['type']
                if node.stay_at_home:
                    num_stay_at_home[household_type] += 1
                if node.isolated:
                    num_isolated[household_type] += 1
                if node.infected:
                    num_infected[household_type] += 1
                if node.recovered:
                    num_recovered[household_type] += 1
                if node.hospitalized:
                    num_hospitalized[household_type] += 1
                if node.hospital_date > 0:
                    num_hospitalized_cumsum[household_type] += 1
                if node.ICU:
                    num_ICU[household_type] += 1
                if node.deceased == 'virus':
                    num_viral_deaths[household_type] += 1
                if node.deceased == 'undertreatment':
                    num_undertreated_deaths[household_type] += 1
                if node.deceased == 'despair':
                    num_deaths_of_despair[household_type] += 1
                if node.active:
                    num_active[household_type] += 1
            self.time_steps_recorded.append(self.time)
            total_output = self.get_total_output()
            for household_type in {'poor', 'rich'}:
                self.stay_at_home_count[household_type].append(num_stay_at_home[household_type])
                self.isolation_count[household_type].append(num_isolated[household_type])
                self.infections[household_type].append(num_infected[household_type])
                self.recoveries[household_type].append(num_recovered[household_type])
                self.hospitalizations[household_type].append(num_hospitalized[household_type])
                self.hospitalizations_cumsum[household_type].append(num_hospitalized_cumsum[household_type])
                self.ICU_count[household_type].append(num_ICU[household_type])
                self.viral_deaths[household_type].append(num_viral_deaths[household_type])
                self.undertreated_deaths[household_type].append(num_undertreated_deaths[household_type])
                self.deaths_of_despair[household_type].append(num_deaths_of_despair[household_type])
                self.active_count[household_type].append(num_active[household_type])
                self.total_output[household_type].append(total_output[household_type])
                self.total_subsidy[household_type].append(total_subsidy[household_type])

    def save_to_csv(self, filename_tag='single_simulation', timestamp=True, path=''):
        """Save aggregate statistics to a csv file"""
        stats = ['stay_at_home_count', 'isolation_count', 'infections', 'recoveries',
                 'hospitalizations', 'hospitalizations_cumsum', 'ICU_count', 'viral_deaths', 'undertreated_deaths',
                 'deaths_of_despair', 'active_count', 'total_output', 'total_subsidy']
        if timestamp:
            filename = filename_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
        else:
            filename = filename_tag + '.csv'
        with open(path + filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            if not self.household_grouping:
                header = ['time'] + stats
                writer.writerow(header)
                for idx in range(len(self.time_steps_recorded)):
                    values = [self.time_steps_recorded[idx]]
                    for x in stats:
                        values.append(self.__getattribute__(x)[idx])
                    writer.writerow(values)
            else:
                header = ['time'] + [i for x in stats for i in [x + '_poor', x + '_rich']]
                writer.writerow(header)
                for idx in range(len(self.time_steps_recorded)):
                    values = [self.time_steps_recorded[idx]]
                    for x in stats:
                        values.extend([self.__getattribute__(x)['poor'][idx], self.__getattribute__(x)['rich'][idx]])
                    writer.writerow(values)

    def get_raw_data(self, stat, household_type=None):
        """Return raw data"""
        if not self.household_grouping:
            assert household_type is None, 'Without household grouping, household type must be None'
        else:
            assert household_type in {None, 'poor', 'rich'}, 'Household type must be None, "poor", or "rich"'
        if not self.household_grouping:
            raw_data = self.__getattribute__(stat)
        elif household_type == 'poor':
            raw_data = self.__getattribute__(stat)['poor']
        elif household_type == 'rich':
            raw_data = self.__getattribute__(stat)['rich']
        else:
            raw_data = list(np.array(self.__getattribute__(stat)['poor'])
                            + np.array(self.__getattribute__(stat)['rich']))
        return raw_data

    def normalize(self, stat, household_type=None):
        """Return a list of normalized values for the statistic

        stat: Name of the statistic as a string

        For a list of counts, each number is divided by the initial population size.
        For total subsidy, each number is divided by the initial total output.
        For other statistics, each number is divided by the first number of the list.
        """
        stats_count = {'stay_at_home_count', 'isolation_count',
                       'infections', 'recoveries', 'hospitalizations', 'hospitalizations_cumsum', 'ICU_count',
                       'viral_deaths', 'undertreated_deaths', 'deaths_of_despair', 'active_count'}
        raw_data = self.get_raw_data(stat, household_type)
        if stat in stats_count:
            if household_type == 'poor':
                num_nodes = self.num_nodes_poor
            elif household_type == 'rich':
                num_nodes = self.num_nodes_rich
            else:
                num_nodes = self.num_nodes
            res = [x / num_nodes for x in raw_data]
        elif stat == 'total_subsidy':
            if not self.household_grouping:
                init_total_output = self.initial_total_output
            elif household_type == 'poor':
                init_total_output = self.initial_total_output['poor']
            elif household_type == 'rich':
                init_total_output = self.initial_total_output['rich']
            else:
                init_total_output = self.initial_total_output['poor'] + self.initial_total_output['rich']
            res = [x / init_total_output for x in raw_data]
        else:
            res = [x / raw_data[0] for x in raw_data]
        return res

    def plot_age_dist(self, households=None,
                      figsize=(6, 4), save=False, filename_tag='age_dist', timestamp=False, path=''):
        """Plot the age distribution for the households specified
        If households is None, then all nodes are considered.
        """
        # noinspection PyTypeChecker
        assert households is None or isinstance(households, (list, tuple)), 'Households must be None, a list, ' \
                                                                            'or a tuple'
        if households is None:
            ages = np.empty(self.num_nodes)
            for i in range(self.num_nodes):
                ages[i] = int(self.node_dict[i].age.split('-')[0])
        else:
            ages = []
            for j in households:
                for i in self.household_dict[j]['members']:
                    ages.append(int(self.node_dict[i].age.split('-')[0]))
            ages = np.array(ages)
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=figsize)
        plt.hist(ages, bins=bins, density=True)
        plt.xlabel('Age')
        plt.ylabel('Density')
        if save:
            if timestamp:
                filename = filename_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
            else:
                filename = filename_tag + '.pdf'
            plt.savefig(path + filename, transparent=True, bbox_inches='tight')
        plt.show()
        plt.pause(3)
        plt.close()

    def plot_p_despair(self, average_worker_degree, p_despair_factor=5.5e-5, x_label='Output',
                       figsize=(6, 4), save=False, filename_tag='p_despair', timestamp=False, path=''):
        """Plot the probability of despair as a function of output or output loss"""
        plt.figure(figsize=figsize)
        plt.style.use('seaborn-white')
        text_color = mcolors.CSS4_COLORS['gray']
        linewidth = 3
        fontsize = 12
        if x_label == 'Output':
            x = np.linspace(Node.personal_output,
                            Node.personal_output + average_worker_degree * self.colleague_linkage_output, num=100)
            y = np.zeros(len(x))
            for i in range(len(x)):
                output_loss = (1 + 1 / self.personal_to_linkage_output_ratio) * Node.personal_output - x[i]
                y[i] = self.sigmoid(output_loss) * p_despair_factor
            text_y_pos = (np.min(y) + np.max(y)) / 4
            plt.plot(x, y, linewidth=linewidth, color=mcolors.CSS4_COLORS['cornflowerblue'])
            plt.axvline(x=np.min(x), linewidth=linewidth/2, linestyle='--', color=text_color)
            plt.text(np.min(x) + .01, text_y_pos, r'$y$', fontsize=fontsize, color=text_color)
            plt.axvline(x=(np.min(x) + np.max(x))/2, linewidth=linewidth/2, linestyle='--', color=text_color)
            plt.text((np.min(x) + np.max(x))/2 - .045, text_y_pos, r'$y+\frac{nx}{2}$', fontsize=fontsize, color=text_color)
            plt.axvline(x=np.max(x), linewidth=linewidth/2, linestyle='--', color=text_color)
            plt.text(np.max(x) - .05, text_y_pos, r'$y+nx$', fontsize=fontsize, color=text_color)
        elif x_label == 'Output loss':
            x = np.linspace(0, average_worker_degree * self.colleague_linkage_output, num=100)
            y = np.zeros(len(x))
            for i in range(len(x)):
                y[i] = self.sigmoid(x[i]) * p_despair_factor
            text_y_pos = (np.min(y) + np.max(y)) / 6
            plt.plot(x, y, linewidth=linewidth, color=mcolors.CSS4_COLORS['cornflowerblue'])
            plt.axvline(x=(np.min(x) + np.max(x)) / 2, linewidth=linewidth / 2, linestyle='--', color=text_color)
            plt.text((np.min(x) + np.max(x)) / 2 + .004, text_y_pos, 'Losing half of\neconomic connections',
                     fontsize=fontsize, color=text_color)
        else:
            raise ValueError('x_label must be either "Output" or "Output loss"')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Probability of death of despair', fontsize=12)
        plt.tick_params(axis='both', labelsize=12)
        if save:
            if timestamp:
                filename = filename_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
            else:
                filename = filename_tag + '.pdf'
            plt.savefig(path + filename, transparent=True, bbox_inches='tight')
        plt.show()
        plt.pause(3)
        plt.close()

    def plot_time_series(self, y_strs, y_labels, title, normalized=True, household_type=None,
                         vlines=None, vline_labels=None, figsize=(6, 4),
                         save=False, filename_tag='time_series', timestamp=False, path=''):
        """Plot a single or multiple time series of recorded statistics"""
        linestyles = ['--', ':', '-.', (0, (1, 10)), (0, (5, 10))]
        if vlines is not None:
            num_vlines = len(vlines)
        else:
            num_vlines = 0
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=figsize)
        plt.title(title)
        for idx, y in enumerate(y_strs):
            assert isinstance(y, str), 'y_strs should be a list of names of y as strings'
            y = y.replace('self.', '', 1)
            if normalized:
                y_values = self.normalize(y, household_type)
            else:
                y_values = self.get_raw_data(y, household_type)
            plt.plot(self.time_steps_recorded, y_values, alpha=.7, label=y_labels[idx])
        plt.xlabel('Day')
        for i in range(num_vlines):
            plt.axvline(x=vlines[i], linestyle=linestyles[i], alpha=.5, color='black', label=vline_labels[i])
        plt.legend(loc='center left', bbox_to_anchor=(1, .5))
        if save:
            title = title.replace(' ', '_')
            if timestamp:
                filename = filename_tag + '_' + title + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
            else:
                filename = filename_tag + '_' + title + '.pdf'
            plt.savefig(path + filename, transparent=True, bbox_inches='tight')
        plt.show()
        plt.pause(3)
        plt.close()

    def plot_econ_time_series(self, title, normalized=False, household_type=None, y_lims=None,
                              vlines=None, vline_labels=None, figsize=(4, 4),
                              save=False, filename_tag='econ_measures_separate',
                              timestamp=False, path=''):
        """Plot all time series of economic measures in one figure"""

        y_strs = ['stay_at_home_count', 'active_count', 'total_output', 'total_subsidy']
        y_labels = ['Stay at home', 'Active', 'Total output', 'Total subsidy']
        colors = [mcolors.CSS4_COLORS['darkgray'], mcolors.CSS4_COLORS['cornflowerblue'],
                  mcolors.CSS4_COLORS['navy'], mcolors.CSS4_COLORS['deepskyblue']]

        linestyles = ['--', ':', '-.', (0, (1, 10)), (0, (5, 10))]
        if vlines is not None:
            num_vlines = len(vlines)
        else:
            num_vlines = 0

        plt.style.use('seaborn-whitegrid')
        fig, axes = plt.subplots(nrows=len(y_strs), figsize=figsize)
        handles = []
        labels = []

        for idx, y in enumerate(y_strs):
            assert isinstance(y, str), 'y_strs should be a list of names of y as strings'
            y = y.replace('self.', '', 1)
            if normalized:
                y_values = self.normalize(y, household_type)
            else:
                y_values = self.get_raw_data(y, household_type)
            if y_lims is not None:
                axes[idx].set_ylim(y_lims[2 * idx], y_lims[2 * idx + 1])
            axes[idx].plot(self.time_steps_recorded, y_values, color=colors[idx], alpha=.7, label=y_labels[idx])
            for i in range(num_vlines):
                axes[idx].axvline(x=vlines[i], linestyle=linestyles[i], alpha=.5, color='black', label=vline_labels[i])
            subplot_handles, subplot_labels = axes[idx].get_legend_handles_labels()
            if idx < (len(y_strs) - 1):
                handles.append(subplot_handles[0])
                labels.append(subplot_labels[0])
                plt.setp(axes[idx].get_xticklabels(), visible=False)  # Make tick labels for x invisible
            else:
                handles.extend(subplot_handles)
                labels.extend(subplot_labels)
        axes[0].set_title(title)
        axes[-1].set(xlabel='Day')
        fig.subplots_adjust(left=.2)
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .5))
        plt.tight_layout()
        if save:
            title = title.replace(' ', '_')
            if timestamp:
                filename = filename_tag + '_' + title + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
            else:
                filename = filename_tag + '_' + title + '.pdf'
            plt.savefig(path + filename, transparent=True, bbox_inches='tight')
        plt.show()
        plt.pause(3)
        plt.close()

    def plot_all_time_series(self, title, normalized=True, household_type=None, y_lims=None,
                             vlines=None, vline_labels=None, figsize=(4, 4),
                             save=False, filename_tag='all_measures', timestamp=False, path=''):
        """Plot all time series recorded in one figure"""

        y_strs_l = ['infections', 'recoveries', 'stay_at_home_count', 'isolation_count',
                    'active_count', 'total_output', 'total_subsidy']
        y_labels_l = ['Infections', 'Recoveries', 'Stay at home', 'Isolation',
                      'Active', 'Total output', 'Total subsidy']
        colors_l = [mcolors.CSS4_COLORS['chocolate'], mcolors.CSS4_COLORS['green'], mcolors.CSS4_COLORS['darkgray'],
                    mcolors.CSS4_COLORS['dimgray'], mcolors.CSS4_COLORS['cornflowerblue'], mcolors.CSS4_COLORS['navy'],
                    mcolors.CSS4_COLORS['deepskyblue']]

        y_strs_s = ['hospitalizations', 'ICU_count', 'viral_deaths', 'undertreated_deaths', 'deaths_of_despair']
        y_labels_s = ['Hospitalizations', 'ICU', 'Deaths from virus', 'Deaths from undertreatment', 'Deaths of despair']
        colors_s = [mcolors.CSS4_COLORS['mediumorchid'], mcolors.CSS4_COLORS['purple'],
                    mcolors.CSS4_COLORS['goldenrod'], mcolors.CSS4_COLORS['gold'], mcolors.CSS4_COLORS['red']]

        linestyles = ['--', ':', '-.', (0, (1, 10)), (0, (5, 10))]
        if vlines is not None:
            num_vlines = len(vlines)
        else:
            num_vlines = 0

        plt.style.use('seaborn-whitegrid')
        fig, (ax_l, ax_s) = plt.subplots(nrows=2, figsize=figsize)

        for idx, y in enumerate(y_strs_l):
            assert isinstance(y, str), 'y_strs should be a list of names of y as strings'
            y = y.replace('self.', '', 1)
            if normalized:
                y_values = self.normalize(y, household_type)
            else:
                y_values = self.get_raw_data(y, household_type)
            if y_lims is not None:
                ax_l.set_ylim(y_lims[2], y_lims[3])
            ax_l.plot(self.time_steps_recorded, y_values, color=colors_l[idx], alpha=.7, label=y_labels_l[idx])
        for i in range(num_vlines):
            ax_l.axvline(x=vlines[i], linestyle=linestyles[i], alpha=.5, color='black', label=vline_labels[i])
        ax_l.set_title(title)
        plt.setp(ax_l.get_xticklabels(), visible=False)  # Make tick labels for x invisible

        for idx, y in enumerate(y_strs_s):
            assert isinstance(y, str), 'y_strs should be a list of names of y as strings'
            y = y.replace('self.', '', 1)
            if normalized:
                y_values = self.normalize(y, household_type)
            else:
                y_values = self.get_raw_data(y, household_type)
            if y_lims is not None:
                ax_s.set_ylim(y_lims[0], y_lims[1])
            ax_s.plot(self.time_steps_recorded, y_values, color=colors_s[idx], alpha=.7, label=y_labels_s[idx])
        for i in range(num_vlines):
            ax_s.axvline(x=vlines[i], linestyle=linestyles[i], alpha=.5, color='black', label=vline_labels[i])
        ax_s.set(xlabel='Day')

        fig.subplots_adjust(left=.2)
        handles_l, labels_l = ax_l.get_legend_handles_labels()
        handles_s, labels_s = ax_s.get_legend_handles_labels()
        if num_vlines == 0:
            handles = handles_l + handles_s
            labels = labels_l + labels_s
        else:
            handles = handles_l[:-num_vlines] + handles_s
            labels = labels_l[:-num_vlines] + labels_s
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .5))
        plt.tight_layout()
        if save:
            title = title.replace(' ', '_')
            if timestamp:
                filename = filename_tag + '_' + title + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
            else:
                filename = filename_tag + '_' + title + '.pdf'
            plt.savefig(path + filename, transparent=True, bbox_inches='tight')
        plt.show()
        plt.pause(3)
        plt.close()


def monte_carlo_save_to_csv(household_network=True, same_age_household=False,
                            num_households=int(2.67e5), max_household_sizes=4,
                            econ_network='watts strogatz', average_worker_degree=20, rewiring_prob=.5,
                            vulnerable_group=True, vulnerable_population_fraction=.01, vulnerability=.05,
                            initial_infection_fraction=1e-3, household_transmission_prob=.25,
                            other_transmission_prob=None,
                            time_steps_pre_lockdown=5, time_steps_post_lockdown_pre_reopening=175,
                            time_steps_post_reopening=0,
                            occupations_staying_at_home='retiree', partial_opening=True,
                            worker_staying_at_home_prob=.4, staying_at_home_output_remaining_scale=1.0,
                            viral_test_fraction_all=1.45e-3, viral_test_fraction_poor=None,
                            contact_tracing_efficacy=.7, subsidy=.1, subsidy_population_factor=.1, allocation=None,
                            hospital_capacity=2.5e-3, undertreatment_effect=.5,
                            despair_prob_factor=5.5e-5, inactive_prob_factor=1e-2,
                            num_trials=10, normalized=True,
                            filename_tag='', timestamp=True, path=''):
    """Perform Monte Carlo simulation and save results to a csv file"""
    assert household_network in {True, False}, 'household_network can be either True or False'
    assert same_age_household in {True, False}, 'same_age_household can be either True or False'
    assert isinstance(num_households, (int, dict)), 'num_households can be either an integer or a dictionary'
    assert type(num_households) is type(max_household_sizes), 'num_households and max_household_sizes must be the ' \
                                                              'same data type '
    assert econ_network in {'watts strogatz', 'fast gnp'}, 'econ_network can be either "watts strogatz" or "fast gnp"'
    assert 0 <= rewiring_prob <= 1, 'rewiring_prob must be between 0 and 1'
    assert vulnerable_group in {True, False}, 'vulnerable_group can be either True or False'
    assert 0 < initial_infection_fraction < 1, 'initial_infection_fraction must be between 0 and 1 exclusive'
    assert time_steps_pre_lockdown >= 0 and time_steps_pre_lockdown == int(time_steps_pre_lockdown), \
        'time_steps_pre_lockdown must be a nonnegative integer'
    assert time_steps_post_lockdown_pre_reopening >= 0 and (time_steps_post_lockdown_pre_reopening ==
                                                            int(time_steps_post_lockdown_pre_reopening)), \
        'time_steps_post_lockdown_pre_reopening must be a nonnegative integer'
    assert time_steps_post_reopening >= 0 and time_steps_post_reopening == int(time_steps_post_reopening), \
        'time_steps_post_reopening must be a nonnegative integer'
    if time_steps_post_lockdown_pre_reopening == 0:
        assert time_steps_post_reopening == 0, 'time_steps_post_reopening must be zero without lockdown'
    if occupations_staying_at_home != '':
        assert set(occupations_staying_at_home.split(',')).issubset({'student', 'worker', 'retiree'}), \
            'occupations_staying_at_home must be an empty string or a string of some of "student", "worker", and ' \
            '"retiree" separated by comma '
    assert partial_opening in {True, False}, 'partial_opening can be either True or False'
    assert 0 <= viral_test_fraction_all <= 1, 'viral_test_fraction_all must be between 0 and 1'
    assert viral_test_fraction_poor is None or 0 <= viral_test_fraction_poor <= 1, 'viral_test_fraction_poor must be ' \
                                                                                   'either None or between 0 and 1 '
    assert num_trials > 0 and num_trials == int(num_trials), 'num_trials must be a positive integer'
    assert normalized in {True, False}, 'normalized must be either True or False'
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    assert timestamp in {True, False}, 'timestamp must be either True or False'
    assert isinstance(path, str), 'path must be a string'

    if other_transmission_prob is None:
        other_transmission_prob = household_transmission_prob / 50

    # Save simulation specifications to a CSV file
    stats = ['stay_at_home_count', 'isolation_count', 'infections', 'recoveries',
             'hospitalizations', 'hospitalizations_cumsum', 'ICU_count', 'viral_deaths', 'undertreated_deaths',
             'deaths_of_despair', 'active_count', 'total_output', 'total_subsidy']
    if timestamp:
        filename = 'monte_carlo_raw_stats' + filename_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
    else:
        filename = 'monte_carlo_raw_stats' + filename_tag + '.csv'
    with open(path + filename, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['household_network', household_network])
        writer.writerow(['same_age_household', same_age_household])
        writer.writerow(['num_households', num_households])
        writer.writerow(['max_household_sizes', max_household_sizes])
        writer.writerow(['econ_network', econ_network])
        writer.writerow(['average_worker_degree', average_worker_degree])
        writer.writerow(['rewiring_prob', rewiring_prob])
        writer.writerow(['initial_infection_fraction', initial_infection_fraction])
        writer.writerow(['household_transmission_prob', household_transmission_prob])
        writer.writerow(['other_transmission_prob', other_transmission_prob])
        writer.writerow(['time_steps_pre_lockdown', time_steps_pre_lockdown])
        writer.writerow(['time_steps_post_lockdown_pre_reopening', time_steps_post_lockdown_pre_reopening])
        writer.writerow(['time_steps_post_reopening', time_steps_post_reopening])
        writer.writerow(['occupations_staying_at_home', occupations_staying_at_home])
        writer.writerow(['partial_opening', partial_opening])
        writer.writerow(['worker_staying_at_home_prob', worker_staying_at_home_prob])
        writer.writerow(['staying_at_home_output_remaining_scale', staying_at_home_output_remaining_scale])
        writer.writerow(['viral_test_fraction_all', viral_test_fraction_all])
        writer.writerow(['viral_test_fraction_poor', viral_test_fraction_poor])
        writer.writerow(['contact_tracing_efficacy', contact_tracing_efficacy])
        writer.writerow(['subsidy', subsidy])
        writer.writerow(['subsidy_population_factor', subsidy_population_factor])
        writer.writerow(['allocation', allocation])
        writer.writerow(['hospital_capacity', hospital_capacity])
        writer.writerow(['undertreatment_effect', undertreatment_effect])
        writer.writerow(['despair_prob_factor', despair_prob_factor])
        writer.writerow(['inactive_prob_factor', inactive_prob_factor])
        if isinstance(num_households, int):
            household_grouping = False
            header = ['trial', 'time'] + stats
        else:
            household_grouping = True
            header = ['trial', 'time'] + [i for x in stats for i in [x, x + '_poor', x + '_rich']]
        writer.writerow(header)

    for trial in tqdm(range(num_trials)):

        # Initialize a new simulation
        sim = Simulation()
        sim.calc_output_measures(average_worker_degree)

        # Generate networks
        if household_network:
            if not household_grouping:
                sim.household_network(num_households, max_household_sizes, None, 1.0, same_age_household)
            else:
                sim.household_network(num_households['poor'], max_household_sizes['poor'], 'poor', 1.0,
                                      same_age_household)
                sim.household_network(num_households['rich'], max_household_sizes['rich'], 'rich', 1.0,
                                      same_age_household)
                sim.calc_rich_to_poor_output_ratio()
        if not sim.household_grouping:
            average_num_nodes = int(num_households * (1 + max_household_sizes) / 2)
        else:
            average_num_nodes = int((num_households['poor'] * (1 + max_household_sizes['poor'])
                                     + num_households['rich'] * (1 + max_household_sizes['rich'])) / 2)
        if econ_network == 'watts strogatz':
            sim.econ_network(watts_strogatz_graph, [average_num_nodes, average_worker_degree, rewiring_prob])
        elif econ_network == 'fast gnp':
            worker_edge_prob = average_worker_degree / average_num_nodes
            sim.econ_network(fast_gnp_random_graph, [average_num_nodes, worker_edge_prob])
        sim.populate_connections()
        if vulnerable_group:
            sim.add_vulnerable_group(vulnerable_population_fraction, vulnerability)

        # Seed initial infections
        num_initial_infections = math.ceil(initial_infection_fraction * sim.num_nodes)
        initial_infections = np.random.choice(sim.num_nodes, num_initial_infections, replace=False)
        sim.seed_simulation(initial_infections, record_stats=True)

        # Calculate testing capacity
        if viral_test_fraction_poor is None:
            num_viral_tests = int(viral_test_fraction_all * sim.num_nodes)
        else:
            rich_fraction = sim.num_nodes_rich / sim.num_nodes
            viral_test_fraction_rich = (viral_test_fraction_all - (1 - rich_fraction)
                                        * viral_test_fraction_poor) / rich_fraction
            assert 0 <= viral_test_fraction_rich <= 1, 'Testing rate for the rich must be between 0 and 1'
            num_viral_tests = {'poor': int(viral_test_fraction_poor * sim.num_nodes_poor),
                               'rich': int(viral_test_fraction_rich * sim.num_nodes_rich)}

        # Configure simulation
        def simulation_step():
            sim.simulation_step(household_transmission_prob, other_transmission_prob,
                                num_viral_tests, contact_tracing_efficacy,
                                hospital_capacity, undertreatment_effect,
                                despair_prob_factor, subsidy, subsidy_population_factor, allocation,
                                inactive_prob_factor, record_stats=True)

        # Pre-lockdown
        for _ in range(time_steps_pre_lockdown):
            simulation_step()

        if time_steps_post_lockdown_pre_reopening > 0:

            # Lockdown
            if occupations_staying_at_home != '':
                sim.stay_at_home_by_occupation_policy(occupations_staying_at_home.split(','),
                                                      staying_at_home_output_remaining_scale)
            if partial_opening:
                sim.partial_opening_policy(worker_staying_at_home_prob, staying_at_home_output_remaining_scale)

            # Post-lockdown and pre-reopening
            for _ in range(time_steps_post_lockdown_pre_reopening):
                simulation_step()

            if time_steps_post_reopening > 0:

                # Reopening
                sim.reopen_policy()

                # Post-reopening
                for _ in range(time_steps_post_reopening):
                    simulation_step()

        # Save aggregate statistics to a CSV file
        num_digits_after_decimal_point = 10
        if not sim.household_grouping:
            if normalized:
                for x in stats:
                    setattr(sim, x, sim.normalize(x))
            with open(path + filename, mode='a') as f:
                writer = csv.writer(f, delimiter=',')
                for idx in range(len(sim.time_steps_recorded)):
                    values = [trial, sim.time_steps_recorded[idx]]
                    for x in stats:
                        if normalized:
                            values.append(round(sim.__getattribute__(x)[idx], num_digits_after_decimal_point))
                        else:
                            values.append(sim.__getattribute__(x)[idx])
                    writer.writerow(values)
        else:
            if normalized:
                for x in stats:
                    setattr(sim, x, {'all': sim.normalize(x), 'poor': sim.normalize(x, 'poor'),
                                     'rich': sim.normalize(x, 'rich')})
            else:
                for x in stats:
                    x_poor = sim.__getattribute__(x)['poor']
                    x_rich = sim.__getattribute__(x)['rich']
                    x_all = [x_poor[i] + x_rich[i] for i in range(len(x_poor))]
                    setattr(sim, x, {'all': x_all, 'poor': x_poor, 'rich': x_rich})
            with open(path + filename, mode='a') as f:
                writer = csv.writer(f, delimiter=',')
                for idx in range(len(sim.time_steps_recorded)):
                    values = [trial, sim.time_steps_recorded[idx]]
                    for x in stats:
                        if normalized:
                            values.extend([round(sim.__getattribute__(x)['all'][idx], num_digits_after_decimal_point),
                                           round(sim.__getattribute__(x)['poor'][idx], num_digits_after_decimal_point),
                                           round(sim.__getattribute__(x)['rich'][idx], num_digits_after_decimal_point)])
                        else:
                            values.extend([sim.__getattribute__(x)['all'][idx], sim.__getattribute__(x)['poor'][idx],
                                           sim.__getattribute__(x)['rich'][idx]])
                    writer.writerow(values)


def monte_carlo(create_dir=True, dir_name_tag='', timestamp=True, path='', **kwargs):
    """Perform Monte Carlo simulation for one or multiple sets of specifications, and save results to
    a directory of csv files
    """
    if create_dir:  # Create a directory for saving simulation statistics
        dir_name = 'monte_carlo_data' + dir_name_tag
        if timestamp:
            dir_name += '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name += '/'
        csv_path = os.path.join(path, dir_name)
        os.mkdir(csv_path)
    else:
        csv_path = path
    kwargs['path'] = csv_path
    iter_vars = []
    iter_values = []
    for var, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            iter_vars.append(var)
            iter_values.append(value)
    if iter_vars:
        for value_tuple in itertools.product(*iter_values):
            for idx in range(len(iter_vars)):
                kwargs[iter_vars[idx]] = value_tuple[idx]
            monte_carlo_save_to_csv(**kwargs)
    else:
        monte_carlo_save_to_csv(**kwargs)
    print('Simulation done!')


def monte_carlo_read_csv(x, x_method, y, y_method, color_by, color_by_method,
                         by_household_type=False, drop_after_time=None, csv_path=''):
    """Read Monte Carlo simulation results from all csv files in the directory specified"""
    assert x_method in {None, 'min', 'max', 'average'}, 'x_method must be None, "min", "max", or "average"'
    assert y_method in {None, 'min', 'max', 'average'}, 'y_method must be None, "min", "max", or "average"'
    assert color_by_method in {None, 'min', 'max', 'average'}, 'color_by_method must be None, "min", "max", ' \
                                                               'or "average" '
    assert by_household_type in {True, False}, 'by_household_type must be either True or False'
    if drop_after_time is not None:
        assert drop_after_time == int(drop_after_time) and drop_after_time >= 0, 'drop_after_time must be None or' \
                                                                                 'a nonnegative integer'
    filenames = glob.glob(csv_path + '*.csv')
    filenames.sort()
    if not by_household_type:
        x_values = np.zeros(len(filenames))
        y_values = np.zeros(len(filenames))
        color_by_values = np.zeros(len(filenames))
        x_std = np.zeros(len(filenames))
        y_std = np.zeros(len(filenames))
        color_by_std = np.zeros(len(filenames))
    else:
        x_values = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
        y_values = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
        color_by_values = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
        x_std = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
        y_std = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
        color_by_std = {'poor': np.zeros(len(filenames)), 'rich': np.zeros(len(filenames))}
    for idx, f in enumerate(filenames):
        df = pd.read_fwf(f, header=None)
        num_specifications = 0
        for i in range(len(df)):
            row = df.iloc[i].values[0].split(',')
            first_entry = row[0]
            if first_entry == x[0] and x_method is None:
                if len(row) > 2:  # Specification of this row was a dictionary
                    dictionary = ast.literal_eval(ast.literal_eval(','.join(row[1:])))
                    if not by_household_type:
                        x_values[idx] = dictionary[x[1]]
                    else:
                        x_values['poor'][idx] = x_values['rich'][idx] = dictionary[x[1]]
                elif not by_household_type:
                    x_values[idx] = row[1]
                else:
                    x_values['poor'][idx] = x_values['rich'][idx] = row[1]
            elif first_entry == y[0] and y_method is None:
                if len(row) > 2:  # Specification of this row was a dictionary
                    dictionary = ast.literal_eval(ast.literal_eval(','.join(row[1:])))
                    if not by_household_type:
                        y_values[idx] = dictionary[y[1]]
                    else:
                        y_values['poor'][idx] = y_values['rich'][idx] = dictionary[y[1]]
                elif not by_household_type:
                    y_values[idx] = row[1]
                else:
                    y_values['poor'][idx] = y_values['rich'][idx] = row[1]
            if first_entry == color_by[0] and color_by_method is None:
                if len(row) > 2:  # Specification of this row was a dictionary
                    dictionary = ast.literal_eval(ast.literal_eval(','.join(row[1:])))
                    if not by_household_type:
                        color_by_values[idx] = dictionary[color_by[1]]
                    else:
                        color_by_values['poor'][idx] = color_by_values['rich'][idx] = dictionary[color_by[1]]
                elif not by_household_type:
                    color_by_values[idx] = row[1]
                else:
                    color_by_values['poor'][idx] = color_by_values['rich'][idx] = row[1]
            if first_entry == 'trial':
                num_specifications = i
                break
        # Drop simulation specifications from the data frame
        df.drop(range(num_specifications), axis=0, inplace=True)
        # Expand the data frame so that there is one statistic per column
        df = df[0].str.split(',', expand=True)
        # Add header
        df.columns = df.iloc[0]
        df.columns.name = None
        df.drop([num_specifications], axis=0, inplace=True)
        # Reset row indices
        df.reset_index(drop=True, inplace=True)
        # Convert data types to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        if df.isnull().values.any():
            print(f'Cannot fully convert the data frame to numeric values:\n{f}')
            print('Number of failed conversion by column:')
            s = df.isnull().sum()
            print(s[s > 0], '\n')
        # Drop data that occurred after the specified time
        if drop_after_time is not None:
            df = df[df['time'] <= drop_after_time]
        if not by_household_type:
            # Calculate some more measures
            df['virus_related_deaths'] = df['viral_deaths'] + df['undertreated_deaths']
            df['total_deaths'] = df['virus_related_deaths'] + df['deaths_of_despair']
            df['infections_cumsum'] = df['infections'] + df['recoveries'] + df['virus_related_deaths']
            df['inactive_count'] = 1 - df['active_count']
            df['loss_in_output'] = 1 - df['total_output']
            recoveries_final = df.loc[df.index[-1], 'recoveries']
            hospitalizations_cumsum_final = df.loc[df.index[-1], 'hospitalizations_cumsum']
            virus_related_deaths_final = df.loc[df.index[-1], 'virus_related_deaths']
            infections_cumsum_final = virus_related_deaths_final + recoveries_final
            if infections_cumsum_final > 0:
                df['infection_hospitalization_ratio'] = hospitalizations_cumsum_final / infections_cumsum_final
                df['infection_fatality_ratio'] = virus_related_deaths_final / infections_cumsum_final
                if hospitalizations_cumsum_final > 0:
                    df['hospitalization_fatality_ratio'] = virus_related_deaths_final / hospitalizations_cumsum_final
                else:
                    df['hospitalization_fatality_ratio'] = 0
            else:
                df['infection_hospitalization_ratio'] = 0
                df['infection_fatality_ratio'] = 0
                df['hospitalization_fatality_ratio'] = 0
            # Summarize specified measures
            if x_method == 'min':
                x_values[idx] = df.groupby('trial')[x[0]].min().mean()
                x_std[idx] = df.groupby('trial')[x[0]].min().std()
            elif x_method == 'max':
                x_values[idx] = df.groupby('trial')[x[0]].max().mean()
                x_std[idx] = df.groupby('trial')[x[0]].max().std()
            elif x_method == 'average':
                x_values[idx] = df.groupby('trial')[x[0]].mean().mean()
                x_std[idx] = df.groupby('trial')[x[0]].mean().std()
            if y_method == 'min':
                y_values[idx] = df.groupby('trial')[y[0]].min().mean()
                y_std[idx] = df.groupby('trial')[y[0]].min().std()
            elif y_method == 'max':
                y_values[idx] = df.groupby('trial')[y[0]].max().mean()
                y_std[idx] = df.groupby('trial')[y[0]].max().std()
            elif y_method == 'average':
                y_values[idx] = df.groupby('trial')[y[0]].mean().mean()
                y_std[idx] = df.groupby('trial')[y[0]].mean().std()
            if color_by_method == 'min':
                color_by_values[idx] = df.groupby('trial')[color_by[0]].min().mean()
                color_by_std[idx] = df.groupby('trial')[color_by[0]].min().mean()
            elif color_by_method == 'max':
                color_by_values[idx] = df.groupby('trial')[color_by[0]].max().mean()
                color_by_std[idx] = df.groupby('trial')[color_by[0]].max().mean()
            elif color_by_method == 'average':
                color_by_values[idx] = df.groupby('trial')[color_by[0]].mean().mean()
                color_by_std[idx] = df.groupby('trial')[color_by[0]].mean().mean()
        else:
            assert 'infections_poor' in df.columns.tolist(), 'Household grouping must be used in simulation'
            for h in ['poor', 'rich']:
                # Calculate some more measures
                df['virus_related_deaths_' + h] = (df['viral_deaths_' + h] + df['undertreated_deaths_' + h])
                df['total_deaths_' + h] = (df['virus_related_deaths_' + h] + df['deaths_of_despair_' + h])
                df['infections_cumsum_' + h] = (df['infections_' + h] + df['recoveries_' + h]
                                                + df['virus_related_deaths_' + h])
                df['inactive_count_' + h] = 1 - df['active_count_' + h]
                df['loss_in_output_' + h] = 1 - df['total_output_' + h]
                recoveries_final = df.loc[df.index[-1], 'recoveries_' + h]
                hospitalizations_cumsum_final = df.loc[df.index[-1], 'hospitalizations_cumsum_' + h]
                virus_related_deaths_final = df.loc[df.index[-1], 'virus_related_deaths_' + h]
                infections_cumsum_final = virus_related_deaths_final + recoveries_final
                if infections_cumsum_final > 0:
                    df['infection_hospitalization_ratio_' + h] = hospitalizations_cumsum_final / infections_cumsum_final
                    df['infection_fatality_ratio_' + h] = virus_related_deaths_final / infections_cumsum_final
                    if hospitalizations_cumsum_final > 0:
                        df['hospitalization_fatality_ratio_' + h] = (virus_related_deaths_final
                                                                     / hospitalizations_cumsum_final)
                    else:
                        df['hospitalization_fatality_ratio_' + h] = 0
                else:
                    df['infection_hospitalization_ratio_' + h] = 0
                    df['infection_fatality_ratio_' + h] = 0
                    df['hospitalization_fatality_ratio_' + h] = 0
                    # Summarize specified measures
                if x_method == 'min':
                    x_values[h][idx] = df.groupby('trial')[x[0] + '_' + h].min().mean()
                    x_std[h][idx] = df.groupby('trial')[x[0] + '_' + h].min().std()
                elif x_method == 'max':
                    x_values[h][idx] = df.groupby('trial')[x[0] + '_' + h].max().mean()
                    x_std[h][idx] = df.groupby('trial')[x[0] + '_' + h].max().std()
                elif x_method == 'average':
                    x_values[h][idx] = df.groupby('trial')[x[0] + '_' + h].mean().mean()
                    x_std[h][idx] = df.groupby('trial')[x[0] + '_' + h].mean().std()
                if y_method == 'min':
                    y_values[h][idx] = df.groupby('trial')[y[0] + '_' + h].min().mean()
                    y_std[h][idx] = df.groupby('trial')[y[0] + '_' + h].min().std()
                elif y_method == 'max':
                    y_values[h][idx] = df.groupby('trial')[y[0] + '_' + h].max().mean()
                    y_std[h][idx] = df.groupby('trial')[y[0] + '_' + h].max().std()
                elif y_method == 'average':
                    y_values[h][idx] = df.groupby('trial')[y[0] + '_' + h].mean().mean()
                    y_std[h][idx] = df.groupby('trial')[y[0] + '_' + h].mean().std()
                if color_by_method == 'min':
                    color_by_values[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].min().mean()
                    color_by_std[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].min().std()
                elif color_by_method == 'max':
                    color_by_values[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].max().mean()
                    color_by_std[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].max().std()
                elif color_by_method == 'average':
                    color_by_values[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].mean().mean()
                    color_by_std[h][idx] = df.groupby('trial')[color_by[0] + '_' + h].mean().std()
    return x_values, x_std, y_values, y_std, color_by_values, color_by_std


def monte_carlo_single_plot(x, x_label, x_method, y, y_label, y_method, std=True, drop_after_time=None,
                            figsize=(6, 4), save=False, filename_tag='', timestamp=False,
                            plot_path='', csv_path=''):
    """Create a single plot using Monte Carlo simulation results from all csv files in the directory specified"""
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    x_values, x_std, y_values, y_std, *_ = monte_carlo_read_csv(x, x_method, y, y_method, [None, None], None,
                                                                False, drop_after_time, csv_path)
    # Plot
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figsize)
    plt.plot(x_values, y_values, color=mcolors.CSS4_COLORS['red'])
    if std:
        plt.fill_between(x_values, y_values - y_std, y_values + y_std, facecolor=mcolors.CSS4_COLORS['red'], alpha=.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()


def monte_carlo_multi_plots(x, x_label, x_method, ys, y_labels, y_methods, y_axis_label,
                            y_scale=1, ylim=None, std=True, by_household_type=False, drop_after_time=None,
                            figsize=(6, 4), legend=True, ordered_legends=False, legend_kwargs=None,
                            save=False, filename_tag='', timestamp=False,
                            plot_path='', csv_path=''):
    """Create multiple plots in one figure using Monte Carlo simulation results from all csv files
    in the directory specified
    """
    assert len(ys) == len(y_labels) == len(y_methods), 'ys, y_labels, and y_methods must be the same length'
    assert legend_kwargs is None or isinstance(legend_kwargs, dict), 'legend_kwargs must be either None or a dictionary'
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    colors = [mcolors.CSS4_COLORS['forestgreen'], mcolors.CSS4_COLORS['dodgerblue'], mcolors.CSS4_COLORS['red'],
              mcolors.CSS4_COLORS['darkorange'], mcolors.CSS4_COLORS['purple'], mcolors.CSS4_COLORS['gray']]
    plt.style.use('seaborn-white')
    plt.figure(figsize=figsize)
    linewidth = 3
    for idx in range(len(ys)):
        x_values, x_std, y_values, y_std, *_ = monte_carlo_read_csv(x, x_method, ys[idx], y_methods[idx],
                                                                    [None, None], None,
                                                                    by_household_type, drop_after_time, csv_path)
        if not by_household_type:
            y_values *= y_scale
            y_std *= y_scale
            plt.plot(x_values, y_values, label=y_labels[idx], color=colors[idx], alpha=.7, linewidth=linewidth)
            if std:
                plt.fill_between(x_values, y_values - y_std, y_values + y_std, facecolor=colors[idx], alpha=.2)
        else:
            x_values = x_values['poor']
            for h in {'poor', 'rich'}:
                y_values[h] *= y_scale
                y_std[h] *= y_scale
            plt.plot(x_values, y_values['poor'], label=y_labels[idx] + ' (poor)',
                     linestyle='-', linewidth=linewidth, color=colors[idx], alpha=.7)
            if 'COVID-19' in y_labels[idx]:
                plt.plot(x_values, y_values['rich'], label=y_labels[idx] + ' (rich)',
                         linestyle='--', linewidth=linewidth, color=colors[idx], alpha=.7)
            if std:
                plt.fill_between(x_values, y_values['poor'] - y_std['poor'], y_values['poor'] + y_std['poor'],
                                 facecolor=colors[idx], alpha=.2)
                if 'COVID-19' in y_labels[idx]:
                    plt.fill_between(x_values, y_values['rich'] - y_std['rich'], y_values['rich'] + y_std['rich'],
                                     facecolor=colors[idx], alpha=.2)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tick_params(axis='both', labelsize=12)
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [3, 2, 0, 1]
        if legend_kwargs is None:
            plt.legend([handles[i] for i in order], [labels[i] for i in order])
        else:
            plt.legend([handles[i] for i in order], [labels[i] for i in order], **legend_kwargs)
    else:
        plt.gca.legend_ = None
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()


def monte_carlo_multi_plots_approx(x, x_label, x_method, ys, y_labels, y_methods,
                                   ylim=None, std=True, by_household_type=False, drop_after_time=None,
                                   figsize=(6, 4), legend_kwargs=None, save=False, filename_tag='', timestamp=False,
                                   plot_path='', csv_path=''):
    """Create multiple plots in one figure using Monte Carlo simulation results from all csv files
    in the directory specified
    """
    assert len(ys) == len(y_labels) == len(y_methods), 'ys, y_labels, and y_methods must be the same length'
    assert legend_kwargs is None or isinstance(legend_kwargs, dict), 'legend_kwargs must be either None or a dictionary'
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    color = mcolors.CSS4_COLORS['darkorange']
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    linewidth = 3
    for idx in range(len(ys)):
        x_values, x_std, y_values, y_std, *_ = monte_carlo_read_csv(x, x_method, ys[idx], y_methods[idx],
                                                                    [None, None], None,
                                                                    by_household_type, drop_after_time, csv_path)
        if not by_household_type:
            plt.plot(x_values, y_values, label=y_labels[idx], color=color, alpha=.7, linewidth=linewidth)
            if std:
                plt.fill_between(x_values, y_values - y_std, y_values + y_std, facecolor=color, alpha=.2)
        else:
            x_values = x_values['poor']
            plt.plot(x_values, y_values['poor'], label=y_labels[idx] + ' (poor)',
                     linestyle='-', linewidth=linewidth, color=color, alpha=.7)
            plt.plot(x_values, y_values['rich'], label=y_labels[idx] + ' (rich)',
                     linestyle='--', linewidth=linewidth, color=color, alpha=.7)
            if std:
                plt.fill_between(x_values, y_values['poor'] - y_std['poor'], y_values['poor'] + y_std['poor'],
                                 facecolor=color, alpha=.2)
                plt.fill_between(x_values, y_values['rich'] - y_std['rich'], y_values['rich'] + y_std['rich'],
                                 facecolor=color, alpha=.2)
    plt.xlabel(x_label, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel('Infection rate', fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Mean-field approximation
    def mean_field_approx_poor(max_household_size_poor):
        n = (1 + max_household_size_poor) / 2  # Average household size of the poor community
        initial_infection_fraction = 1e-3
        leave_home_prob = .63
        average_worker_degree = 20
        other_transmission_prob = 5e-3
        secondary_coeff = .25
        total_days = 44
        days_per_step = 11
        exponent = total_days / days_per_step + 1
        base = leave_home_prob ** 2 * n * average_worker_degree * other_transmission_prob * days_per_step
        base *= 1 + base * secondary_coeff
        res = n * initial_infection_fraction * (base ** exponent - 1) / (base - 1)

        time_steps = int(total_days / days_per_step)
        res_damped = n * initial_infection_fraction
        for t in range(1, time_steps + 1):
            factor = (
                             1 - res_damped) * leave_home_prob ** 2 * n * average_worker_degree * other_transmission_prob * days_per_step
            res_damped += n * initial_infection_fraction * (factor * (1 + secondary_coeff * factor)) ** t
        return res, res_damped

    theory_x = np.arange(3, 9, 1)
    theory_y_poor = {'mfa': np.zeros(len(theory_x)), 'damped': np.zeros(len(theory_x))}
    for i in range(len(theory_x)):
        theory_y_poor['mfa'][i], theory_y_poor['damped'][i] = mean_field_approx_poor(theory_x[i])
    plt.plot(theory_x, theory_y_poor['damped'], label='Approximation (poor)',
             color='black', alpha=.7, linewidth=linewidth)
    theory_y_rich = np.array([theory_y_poor['damped'][i] * 4 / (1 + theory_x[i]) for i in range(len(theory_x))])
    plt.plot(theory_x, theory_y_rich, label='Approximation (rich)',
             color='black', alpha=.7, linestyle='--', linewidth=linewidth)
    if legend_kwargs is None:
        plt.legend()
    else:
        plt.legend(**legend_kwargs)
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()


def monte_carlo_multi_controls(x, x_label, y, y_label, y_method, control, control_label,
                               y_scale=1, ylim=None, std=True, by_household_type=False, drop_after_time=None,
                               figsize=(6, 4), legend_kwargs=None, save=False, filename_tag='', timestamp=False,
                               plot_path='', csv_path=''):
    """Create multiple plots in one figure, with each plot corresponding to one control, using Monte Carlo simulation
    results from all csv files in the directory specified
    """
    assert legend_kwargs is None or isinstance(legend_kwargs, dict), 'legend_kwargs must be either None or a dictionary'
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    color = mcolors.CSS4_COLORS['purple']
    plt.style.use('seaborn-white')
    plt.figure(figsize=figsize)
    linewidth = 3
    x_values, _, y_values, y_std, control_values, _ = monte_carlo_read_csv(x, None, y, y_method, control, None,
                                                                           by_household_type, drop_after_time, csv_path)
    if not by_household_type:
        y_values *= y_scale
        y_std *= y_scale
        control_dict = {}
        for c in np.unique(control_values):
            control_dict[c] = {'x_values': [], 'y_values': [], 'y_std': []}
        n_controls = len(control_dict)
        alphas = np.linspace(1 / n_controls, 1, n_controls)
        for idx, c in enumerate(control_values):
            control_dict[c]['x_values'].append(x_values[idx])
            control_dict[c]['y_values'].append(y_values[idx])
            control_dict[c]['y_std'].append(y_std[idx])
        for idx, c in enumerate(np.unique(control_values)):
            label = f'{control_label} {c}'
            plt.plot(control_dict[c]['x_values'], control_dict[c]['y_values'], label=label,
                     color=color, alpha=alphas[idx], linewidth=linewidth)
            if std:
                lower = np.array(control_dict[c]['y_values']) - np.array(control_dict[c]['y_std'])
                upper = np.array(control_dict[c]['y_values']) + np.array(control_dict[c]['y_std'])
                plt.fill_between(control_dict[c]['x_values'], lower, upper, facecolor=color, alpha=.1)
    else:
        for h in {'poor', 'rich'}:
            y_values[h] *= y_scale
            y_std[h] *= y_scale
        control_dict = {}
        for c in np.unique(control_values['poor']):
            control_dict[c] = {'x_values': {'poor': [], 'rich': []},
                               'y_values': {'poor': [], 'rich': []},
                               'y_std': {'poor': [], 'rich': []}
                               }
        n_controls = len(control_dict)
        alphas = np.linspace(1 / n_controls, 1, n_controls)
        for idx, c in enumerate(control_values['poor']):
            for household_type in {'poor', 'rich'}:
                control_dict[c]['x_values'][household_type].append(x_values[household_type][idx])
                control_dict[c]['y_values'][household_type].append(y_values[household_type][idx])
                control_dict[c]['y_std'][household_type].append(y_std[household_type][idx])
        for idx, c in enumerate(np.unique(control_values['poor'])):
            label = f'{control_label} {c}'
            plt.plot(control_dict[c]['x_values']['poor'], control_dict[c]['y_values']['poor'],
                     label=f'{label} (poor)', linestyle='-', linewidth=linewidth, color=color, alpha=alphas[idx])
            if idx == (n_controls - 1):
                plt.plot(control_dict[c]['x_values']['rich'], control_dict[c]['y_values']['rich'],
                         label=f'{label} (rich)', linestyle='--', linewidth=linewidth, color=color, alpha=alphas[idx])
            if std:
                lower = np.array(control_dict[c]['y_values']['poor']) - np.array(control_dict[c]['y_std']['poor'])
                upper = np.array(control_dict[c]['y_values']['poor']) + np.array(control_dict[c]['y_std']['poor'])
                plt.fill_between(control_dict[c]['x_values']['poor'], lower, upper, facecolor=color, alpha=.1)
                if idx == (n_controls - 1):
                    lower = np.array(control_dict[c]['y_values']['rich']) - np.array(control_dict[c]['y_std']['rich'])
                    upper = np.array(control_dict[c]['y_values']['rich']) + np.array(control_dict[c]['y_std']['rich'])
                    plt.fill_between(control_dict[c]['x_values']['rich'], lower, upper, facecolor=color, alpha=.1)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tick_params(axis='both', labelsize=12)
    if legend_kwargs is None:
        plt.legend()
    else:
        plt.legend(**legend_kwargs)
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()


def monte_carlo_plot_cbar(x, x_label, x_method, y, y_label, y_method, color_by, color_by_label,
                          by_household_type=False, drop_after_time=None,
                          figsize=(6, 4), save=False, filename_tag='', timestamp=False,
                          plot_path='', csv_path=''):
    """Create a single plot with a color bar using Monte Carlo simulation results from all csv files
    in the directory specified
    """
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figsize)
    x_values, _, y_values, _, color_by_values, _ = monte_carlo_read_csv(x, x_method, y, y_method, color_by, None,
                                                                        by_household_type, drop_after_time, csv_path)
    if not by_household_type:
        plt.plot(x_values, y_values, label=color_by_label, color='black', zorder=1)
        plt.scatter(x_values, y_values, c=color_by_values, cmap='Reds', s=60, edgecolors='black', zorder=2)
    else:
        color_by_values = color_by_values['poor']
        plt.plot(x_values['poor'], y_values['poor'], label='Poor', color='black', zorder=1)
        plt.scatter(x_values['poor'], y_values['poor'], c=color_by_values,
                    cmap='Reds', s=60, edgecolors='black', zorder=2)
        plt.plot(x_values['rich'], y_values['rich'], label='Rich', color=mcolors.CSS4_COLORS['gray'], zorder=1)
        plt.scatter(x_values['rich'], y_values['rich'], c=color_by_values,
                    cmap='Reds', s=60, edgecolors=mcolors.CSS4_COLORS['gray'], zorder=2)
        plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(color_by_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()


def monte_carlo_contourf(x, x_label, y, y_label, color_by, color_by_label, color_by_method, drop_after_time=None,
                         figsize=(4, 4), save=False, filename_tag='', timestamp=False,
                         plot_path='', csv_path=''):
    """Plot contours using Monte Carlo simulation results from all csv files in the directory specified"""
    assert isinstance(filename_tag, str), 'filename_tag must be a string'
    x_values, _, y_values, _, color_by_values, _ = monte_carlo_read_csv(x, None, y, None, color_by, color_by_method,
                                                                        False, drop_after_time, csv_path)
    color_by_dict = {}
    for i in range(len(x_values)):
        color_by_dict[(x_values[i], y_values[i])] = color_by_values[i]
    x_values_unique = np.unique(x_values)
    y_values_unique = np.unique(y_values)
    xx, yy = np.meshgrid(x_values_unique, y_values_unique)
    z = np.zeros((len(y_values_unique), len(x_values_unique)))
    for col in range(len(x_values_unique)):
        for row in range(len(y_values_unique)):
            if (x_values_unique[col], y_values_unique[row]) in color_by_dict:
                z[row, col] = color_by_dict[(x_values_unique[col], y_values_unique[row])]
            else:
                z[row, col] = np.nan
    # Plot
    plt.style.use('seaborn-white')
    plt.figure(figsize=figsize)
    h = plt.contourf(xx, yy, z, 20, cmap='Oranges')
    plt.colorbar(h)
    plt.title(color_by_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        if timestamp:
            filename = 'monte_carlo_' + filename_tag + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf'
        else:
            filename = 'monte_carlo' + filename_tag + '.pdf'
        plt.savefig(plot_path + filename, transparent=True, bbox_inches='tight')
    plt.show()
    plt.pause(3)
    plt.close()
