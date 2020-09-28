import math
import pytest
import random
import numpy as np
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph, watts_strogatz_graph

from src.models import sample_age, Node, Simulation


class TestNode:
    def test_infected_update_day_deceased(self):
        age = sample_age()
        node = Node(age)
        node.active = True
        node.output = 12345
        node.infected = True

        node.symptom_date = 1
        node.hospital_date = 2
        node.ICU_date = 3
        node.death_date = 4
        node.disease_length = 3

        status = node.infected_update_day()
        assert node.days_infected == 1
        assert node.isolated
        assert not node.hospitalized
        assert not node.undertreated
        assert not node.ICU
        assert not node.deceased
        assert not node.recovered
        assert status == 'symptomatic'

        status = node.infected_update_day()
        assert node.days_infected == 2
        assert not node.isolated
        assert node.hospitalized
        assert not node.undertreated
        assert not node.ICU
        assert not node.deceased
        assert not node.recovered
        assert not node.active
        assert node.output == 0
        assert status == 'hospitalized'

        status = node.infected_update_day()
        assert node.days_infected == 3
        assert not node.isolated
        assert node.hospitalized
        assert not node.undertreated
        assert node.ICU
        assert not node.deceased
        assert not node.recovered
        assert not node.active
        assert node.output == 0
        assert status == 'ICU'

        node.undertreated = True

        status = node.infected_update_day()
        assert node.days_infected == 4
        assert not node.isolated
        assert not node.hospitalized
        assert node.undertreated
        assert not node.ICU
        assert node.deceased
        assert not node.recovered
        assert not node.active
        assert node.output == 0
        assert not node.old_active
        assert node.old_output == 0
        assert status == 'undertreatment'

    def test_infected_update_day_recovered_after_ICU_previously_inactive(self):
        age = sample_age()
        node = Node(age)
        node.active = False
        node.output = 0
        node.old_active = False
        node.old_output = 0
        node.infected = True
        node.stay_at_home = True

        node.symptom_date = 1
        node.hospital_date = 1
        node.ICU_date = 1
        node.death_date = 0
        node.disease_length = 2

        status = node.infected_update_day()
        assert node.days_infected == 1
        assert not node.stay_at_home
        assert not node.isolated
        assert node.hospitalized
        assert node.ICU
        assert not node.deceased
        assert not node.recovered
        assert not node.active
        assert node.output == 0
        assert status == 'ICU'

        status = node.infected_update_day()
        assert node.days_infected == 2
        assert not node.stay_at_home
        assert not node.isolated
        assert not node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert node.recovered
        assert not node.active
        assert node.output == 0
        assert status == 'recovered'

    def test_infected_update_day_recovered_before_ICU_previously_active(self):
        age = sample_age()
        node = Node(age)
        node.infected = True
        node.active = True
        node.output = 12345
        node.old_active = node.active
        node.old_output = node.output

        node.symptom_date = 1
        node.hospital_date = 1
        node.ICU_date = 0
        node.death_date = 0
        node.disease_length = 2

        status = node.infected_update_day()
        assert node.days_infected == 1
        assert not node.isolated
        assert node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert not node.recovered
        assert not node.active
        assert node.output == 0
        assert node.old_active
        assert node.old_output == 12345
        assert status == 'hospitalized'

        status = node.infected_update_day()
        assert node.days_infected == 2
        assert not node.isolated
        assert not node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert node.recovered
        assert node.active
        assert node.output == 12345
        assert status == 'recovered'

    def test_infected_update_day_recovered_before_hospitalization_previously_active(self):
        age = sample_age()
        node = Node(age)
        node.infected = True
        node.active = True
        node.output = 12345
        node.old_active = node.active
        node.old_output = node.output

        node.symptom_date = 1
        node.hospital_date = 0
        node.ICU_date = 0
        node.death_date = 0
        node.disease_length = 1

        status = node.infected_update_day()
        assert node.days_infected == 1
        assert not node.isolated
        assert not node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert node.recovered
        assert node.active
        assert node.output == 12345
        assert status == 'recovered'

    def test_infected_update_day_asymptomatic_previously_active(self):
        age = sample_age()
        node = Node(age)
        node.infected = True
        node.active = True
        node.output = 12345
        node.old_active = node.active
        node.old_output = node.output

        node.symptom_date = 0
        node.hospital_date = 0
        node.ICU_date = 0
        node.death_date = 0
        node.disease_length = 2

        status = node.infected_update_day()
        assert node.days_infected == 1
        assert not node.isolated
        assert not node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert not node.recovered
        assert node.active
        assert node.output == 12345
        assert status == 'asymptomatic'

        status = node.infected_update_day()
        assert node.days_infected == 2
        assert not node.isolated
        assert not node.hospitalized
        assert not node.ICU
        assert not node.deceased
        assert node.recovered
        assert node.active
        assert node.output == 12345
        assert status == 'recovered'


class TestSimulation:
    def test_init_data_types(self):
        time_series = ['test_backlog', 'stay_at_home_count', 'isolation_count', 'infections', 'recoveries',
                       'hospitalizations', 'hospitalizations_cumsum', 'ICU_count', 'viral_deaths',
                       'undertreated_deaths', 'deaths_of_despair', 'active_count', 'total_output', 'total_subsidy']
        sim = Simulation()
        sim.household_grouping = False
        sim.init_data_types()
        assert sim.initial_total_output == 0
        sim.initial_total_output = None
        for x in time_series:
            assert isinstance(sim.__getattribute__(x), list)
            assert not sim.__getattribute__(x)
            setattr(sim, x, None)
        sim.household_grouping = True
        sim.init_data_types()
        assert isinstance(sim.initial_total_output, dict)
        assert sim.initial_total_output.keys() | set() == {'poor', 'rich'}
        assert sim.initial_total_output['poor'] == 0
        assert sim.initial_total_output['rich'] == 0
        for x in time_series:
            assert isinstance(sim.__getattribute__(x), dict)
            assert sim.__getattribute__(x).keys() | set() == {'poor', 'rich'}
            for household_type in {'poor', 'rich'}:
                assert isinstance(sim.__getattribute__(x)[household_type], list)
                assert not sim.__getattribute__(x)[household_type]

    def test_populate_connections(self):
        sim = Simulation()
        sim.num_nodes = 3
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.node_dict[0].colleagues = [2, 1]
        sim.node_dict[1].colleagues = [0]
        sim.populate_connections()
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].connections == sim.node_dict[i].colleagues

    def test_household_network_no_household_type(self):
        sim = Simulation()
        with pytest.raises(AssertionError, match='Number of households must be a positive integer'):
            sim.household_network(-1, 5, None, 1.0)
        with pytest.raises(AssertionError, match='Number of households must be a positive integer'):
            sim.household_network(1.1, 5, None, 1.0)
        with pytest.raises(AssertionError, match='Maximum household size must be a positive integer'):
            sim.household_network(5, -1, None, 1.0)
        with pytest.raises(AssertionError, match='Maximum household size must be a positive integer'):
            sim.household_network(5, 1.1, None, 1.0)
        with pytest.raises(AssertionError, match='Household type must be None, "poor", or "rich"'):
            sim.household_network(3, 5, 'unknown', 1.0)
        num_households = 4
        max_household_size = 6
        sim.household_network(num_households, max_household_size, None, 1.0)
        assert sim.num_nodes_poor == 0
        assert sim.num_nodes_rich == 0
        assert not sim.household_grouping
        assert sim.max_household_sizes.keys() | set() == {None}
        assert sim.max_household_sizes[None] == max_household_size
        assert isinstance(sim.test_backlog, list)
        assert not sim.test_backlog
        assert sim.initial_total_output == 0
        for i in range(sim.num_nodes):
            household = sim.household_dict[sim.node_dict[i].household]
            assert household['type'] is None
            assert i in household['members']

    def test_household_network_with_household_types(self):
        sim = Simulation()
        num_households_poor, num_households_rich = 4, 7
        max_household_size_poor, max_household_size_rich = 6, 1
        sim.household_network(num_households_poor, max_household_size_poor, 'poor', 1.0)
        assert sim.num_nodes_poor > 0
        assert sim.num_nodes_rich == 0
        assert sim.household_grouping
        assert sim.max_household_sizes.keys() | set() == {'poor'}
        assert sim.max_household_sizes['poor'] == max_household_size_poor
        assert len(sim.household_dict) == num_households_poor
        assert isinstance(sim.test_backlog, dict)
        assert isinstance(sim.initial_total_output, dict)
        for household_type in {'poor', 'rich'}:
            assert isinstance(sim.test_backlog[household_type], list)
            assert not sim.test_backlog[household_type]
            assert sim.initial_total_output[household_type] == 0
        for i in range(sim.num_nodes):
            household = sim.household_dict[sim.node_dict[i].household]
            assert household['type'] == 'poor'
            assert i in household['members']
        sim.household_network(num_households_rich, max_household_size_rich, 'rich', 1.0)
        assert sim.num_nodes_poor > 0
        assert sim.num_nodes_rich > 0
        assert sim.num_nodes == sim.num_nodes_poor + sim.num_nodes_rich
        assert sim.household_grouping
        assert sim.max_household_sizes.keys() | set() == {'poor', 'rich'}
        assert sim.max_household_sizes['poor'] == max_household_size_poor
        assert sim.max_household_sizes['rich'] == max_household_size_rich
        assert len(sim.household_dict) == num_households_poor + num_households_rich
        assert isinstance(sim.test_backlog, dict)
        assert isinstance(sim.initial_total_output, dict)
        for household_type in {'poor', 'rich'}:
            assert isinstance(sim.test_backlog[household_type], list)
            assert not sim.test_backlog[household_type]
            assert sim.initial_total_output[household_type] == 0
        for i in range(sim.num_nodes):
            household = sim.household_dict[sim.node_dict[i].household]
            assert household['type'] in {'poor', 'rich'}
            assert i in household['members']

    def test_add_dict_employed_idx_to_node_idx(self):
        sim = Simulation()
        sim.num_nodes = 5
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        employed = [0, 1, 4]
        for i in range(sim.num_nodes):
            if i in employed:
                sim.node_dict[i].active = True
            else:
                sim.node_dict[i].active = False
        sim.add_dict_active_idx_to_node_idx()
        assert len(sim.active_idx_to_node_idx) == len(employed)
        for idx in range(len(employed)):
            assert sim.active_idx_to_node_idx[idx] == employed[idx]

    def test_graph_function_input_returns_network_fast_gnp(self):
        """Test the case where the input function returns a network"""
        n = 10
        p = 0.6
        seed = random.randint(1, 500)
        g = fast_gnp_random_graph(n, p, seed)
        a = nx.convert_matrix.to_numpy_array(g)
        sim = Simulation()
        sim.num_nodes = n
        sim.node_dict = {i: Node(sample_age()) for i in range(n)}
        with pytest.raises(AssertionError, match='func must be an undirected network'):
            sim.graph_function(fast_gnp_random_graph, [n, p, seed, True], None)  # Directed network
        # No edge type
        sim.graph_function(fast_gnp_random_graph, [n, p, seed], None)  # Undirected network
        for i in range(sim.num_nodes):
            for j in range(sim.num_nodes):
                if j in sim.node_dict[i].connections:
                    assert a[i, j] == 1
                else:
                    assert a[i, j] == 0
        # Economic edges
        num_employed = 0
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                num_employed += 1
        sim.graph_function(fast_gnp_random_graph, [num_employed, p, seed], 'econ')  # Undirected network
        node_idx_to_employed_idx = {value: key for key, value in sim.active_idx_to_node_idx.items()}
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                for j in range(sim.num_nodes):
                    if sim.node_dict[j].active:
                        if j in sim.node_dict[i].colleagues:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 1
                        else:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 0
            else:
                assert sim.node_dict[i].colleagues == []

    def test_graph_function_input_returns_matrix_fast_gnp(self):
        """Test the case where the input function returns an adjacency matrix"""
        n = 20
        p = 0.6
        seed = random.randint(1, 500)
        g = fast_gnp_random_graph(n, p, seed)
        a = nx.convert_matrix.to_numpy_array(g)
        sim = Simulation()
        sim.num_nodes = n
        sim.node_dict = {i: Node(sample_age()) for i in range(n)}
        # No edge type
        sim.graph_function(nx.convert_matrix.to_numpy_array, [g], None)
        for i in range(sim.num_nodes):
            for j in range(sim.num_nodes):
                if j in sim.node_dict[i].connections:
                    assert a[i, j] == 1
                else:
                    assert a[i, j] == 0
        # Economic edges
        sim.graph_function(nx.convert_matrix.to_numpy_array, [g], 'econ')  # Undirected network
        node_idx_to_employed_idx = {value: key for key, value in sim.active_idx_to_node_idx.items()}
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                for j in range(sim.num_nodes):
                    if sim.node_dict[j].active:
                        if j in sim.node_dict[i].colleagues:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 1
                        else:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 0
            else:
                assert sim.node_dict[i].colleagues == []
        with pytest.raises(AssertionError, match='Output of func must be a square matrix'):
            def list_1d():
                return [0, 1]

            sim.graph_function(list_1d, [])
        with pytest.raises(AssertionError, match='Output of func must be a square matrix'):
            def np_array_2d():
                return np.array([[0, 1], [2]])

            sim.graph_function(np_array_2d, [])
        with pytest.raises(AssertionError, match='Adjacency matrix must be symmetric'):
            g = nx.DiGraph()
            g.add_edge(1, 2)
            sim.graph_function(nx.convert_matrix.to_numpy_array, [g])
        with pytest.raises(AssertionError, match='Adjacency matrix must be binary'):
            g = nx.Graph()
            g.add_edge(1, 2, weight=3)
            sim.graph_function(nx.convert_matrix.to_numpy_array, [g])
        with pytest.raises(TypeError, match='a matrix or a network'):
            def not_matrix_or_network():
                return 0

            sim.graph_function(not_matrix_or_network, [])

    def test_graph_function_input_returns_network_watts_strogatz(self):
        n = 10
        k = 2
        p = 0.8
        seed = random.randint(1, 500)
        g = watts_strogatz_graph(n, k, p, seed)
        a = nx.convert_matrix.to_numpy_array(g)
        sim = Simulation()
        sim.num_nodes = n
        sim.node_dict = {i: Node(sample_age()) for i in range(n)}
        # No edge type
        sim.graph_function(watts_strogatz_graph, [n, k, p, seed], None)  # Undirected network
        for i in range(sim.num_nodes):
            for j in range(sim.num_nodes):
                if j in sim.node_dict[i].connections:
                    assert a[i, j] == 1
                else:
                    assert a[i, j] == 0
        # Economic edges
        num_employed = 0
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                num_employed += 1
        sim.graph_function(watts_strogatz_graph, [num_employed, k, p, seed], 'econ')  # Undirected network
        a = nx.convert_matrix.to_numpy_array(watts_strogatz_graph(num_employed, k, p, seed))
        node_idx_to_employed_idx = {value: key for key, value in sim.active_idx_to_node_idx.items()}
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                for j in range(sim.num_nodes):
                    if sim.node_dict[j].active:
                        if j in sim.node_dict[i].colleagues:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 1
                        else:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 0
            else:
                assert sim.node_dict[i].colleagues == []

    def test_graph_function_input_returns_matrix_watts_strogatz(self):
        n = 10
        k = 2
        p = 0.6
        seed = random.randint(1, 500)
        g = watts_strogatz_graph(n, k, p, seed)
        a = nx.convert_matrix.to_numpy_array(g)
        sim = Simulation()
        sim.num_nodes = n
        sim.node_dict = {i: Node(sample_age()) for i in range(n)}
        sim.graph_function(nx.convert_matrix.to_numpy_array, [g])
        # No edge type
        sim.graph_function(nx.convert_matrix.to_numpy_array, [g], None)
        for i in range(sim.num_nodes):
            for j in range(sim.num_nodes):
                if j in sim.node_dict[i].connections:
                    assert a[i, j] == 1
                else:
                    assert a[i, j] == 0
        # Economic edges
        sim.graph_function(nx.convert_matrix.to_numpy_array, [g], 'econ')  # Undirected network
        node_idx_to_employed_idx = {value: key for key, value in sim.active_idx_to_node_idx.items()}
        for i in range(sim.num_nodes):
            if sim.node_dict[i].active:
                for j in range(sim.num_nodes):
                    if sim.node_dict[j].active:
                        if j in sim.node_dict[i].colleagues:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 1
                        else:
                            assert a[node_idx_to_employed_idx[i], node_idx_to_employed_idx[j]] == 0
            else:
                assert sim.node_dict[i].colleagues == []

    def test_econ_network(self):
        n = 20
        k = 2
        p = 0.8
        seed = random.randint(1, 500)
        # No network constructed
        sim = Simulation()
        with pytest.raises(AssertionError, match='No network exists'):
            sim.econ_network(watts_strogatz_graph, [None, k, p, seed])
        with pytest.raises(AssertionError, match='positive integer'):
            sim.econ_network(watts_strogatz_graph, [-1, k, p, seed])
        with pytest.raises(AssertionError, match='positive integer'):
            sim.econ_network(watts_strogatz_graph, [1.5, k, p, seed])
        sim.econ_network(watts_strogatz_graph, [n, k, p, seed])
        assert sim.num_nodes == n
        assert len(sim.node_dict) == n
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if not node.active:
                assert node.colleagues == []
        # A network already exists
        sim = Simulation()
        sim.household_network(4, 3)
        sim.econ_network(watts_strogatz_graph, [n, k, p, seed])
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if not node.active:
                assert node.colleagues == []

    def test_add_vulnerable_group_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 1000
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_grouping = False
        with pytest.raises(AssertionError, match='Fraction of vulnerable population must be between 0 and 1 exclusive'):
            sim.add_vulnerable_group(-0.1, .1)
        with pytest.raises(AssertionError, match='Fraction of vulnerable population must be between 0 and 1 exclusive'):
            sim.add_vulnerable_group(1.1, .1)
        with pytest.raises(AssertionError, match='Vulnerability must be positive'):
            sim.add_vulnerable_group(.1, 0)
        with pytest.raises(AssertionError, match='Vulnerability need to be sufficiently low'):
            sim.add_vulnerable_group(.1, .5)
        with pytest.raises(AssertionError, match='Fraction of vulnerable population and vulnerability'):
            sim.add_vulnerable_group(.999, 1e-2)
        population_fraction, vulnerability = .5, .2
        sim.add_vulnerable_group(population_fraction, vulnerability)
        assert sim.vulnerable_population_fraction == population_fraction
        assert sim.vulnerability_indices['vulnerable'] == 1 + vulnerability
        assert sim.vulnerability_indices['normal'] == 1 - vulnerability * population_fraction / (
                1 - population_fraction)
        vulnerable_count = 0
        for i in range(sim.num_nodes):
            if sim.node_dict[i].vulnerable:
                vulnerable_count += 1
        assert abs(vulnerable_count / sim.num_nodes - population_fraction) < 5e-2

    def test_add_vulnerable_group_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 1000
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_grouping = True
        sim.household_dict = {0: {'type': 'poor', 'members': {}}, 1: {'type': 'rich', 'members': {}}}
        for i in range(sim.num_nodes):
            if i < (sim.num_nodes / 2):
                sim.node_dict[i].household = 0
                sim.num_nodes_poor += 1
            else:
                sim.node_dict[i].household = 1
                sim.num_nodes_rich += 1
        with pytest.raises(AssertionError, match='must be positive and no more than the fraction of the poor'):
            sim.add_vulnerable_group(.9, .1)
        population_fraction, vulnerability = .5, 1e-3
        sim.add_vulnerable_group(population_fraction, vulnerability)
        assert sim.vulnerable_population_fraction == population_fraction
        assert sim.vulnerability_indices['vulnerable'] == 1 + vulnerability
        assert sim.vulnerability_indices['normal'] == 1 - vulnerability * population_fraction / (
                1 - population_fraction)
        vulnerable_count = 0
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.vulnerable:
                assert sim.household_dict[node.household]['type'] == 'poor'
                vulnerable_count += 1
        assert abs(vulnerable_count / sim.num_nodes - population_fraction) < 5e-2

    def test_calc_rich_to_poor_output_ratio(self):
        sim = Simulation()
        sim.household_grouping = True
        sim.num_nodes_poor = 9
        sim.num_nodes_rich = 1
        sim.num_nodes = sim.num_nodes_poor + sim.num_nodes_rich
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': 'rich', 'members': {0}},
                              1: {'type': 'poor', 'members': {1, 2, 3}},
                              2: {'type': 'poor', 'members': {4}},
                              3: {'type': 'poor', 'members': {5, 6, 7, 8, 9}}}
        sim.node_dict[0].household = 0
        for i in {1, 2, 3}:
            sim.node_dict[i].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        personal_output = random.random()
        for i in range(sim.num_nodes):
            sim.node_dict[i].active = True
            sim.node_dict[i].output = personal_output
        sim.total_output_rich_fraction = .45
        sim.calc_rich_to_poor_output_ratio()
        assert (sim.rich_to_poor_output_ratio - 81 / 11) < 1e-5
        assert sim.node_dict[0].output == (personal_output * sim.rich_to_poor_output_ratio)
        for i in range(1, sim.num_nodes):
            assert sim.node_dict[i].output == personal_output

    def test_stay_at_home_by_occupation_policy(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        with pytest.raises(AssertionError, match='The scale of remaining output must be between 0 and 1'):
            sim.stay_at_home_by_occupation_policy(['student'], -0.1)
        with pytest.raises(AssertionError, match='The scale of remaining output must be between 0 and 1'):
            sim.stay_at_home_by_occupation_policy(['student'], 1.1)
        # Staying at home does not affect output
        sim.stay_at_home_by_occupation_policy(['student'], 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation == 'student':
                assert node.stay_at_home
                if node.old_active:
                    assert node.active
                    assert node.output == node.old_output
                else:
                    assert not node.active
                    assert node.output == 0
            else:
                assert not node.stay_at_home
            node.stay_at_home = False
        # Staying at home halves the output
        output_remaining_scale = 0.5
        sim.stay_at_home_by_occupation_policy(['retiree', 'student'], output_remaining_scale)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation in {'retiree', 'student'}:
                assert node.stay_at_home
                if node.old_active:
                    assert node.active
                    assert node.output == node.old_output * output_remaining_scale
                else:
                    assert not node.active
                    assert node.output == 0
            else:
                assert not node.stay_at_home

    def test_partial_opening_policy(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        with pytest.raises(AssertionError, match='The probability of staying at home must be between 0 and 1'):
            sim.partial_opening_policy(-0.1, 0)
        with pytest.raises(AssertionError, match='The probability of staying at home must be between 0 and 1'):
            sim.partial_opening_policy(1.1, 0)
        with pytest.raises(AssertionError, match='The scale of remaining output must be between 0 and 1'):
            sim.partial_opening_policy(0, -0.1)
        with pytest.raises(AssertionError, match='The scale of remaining output must be between 0 and 1'):
            sim.partial_opening_policy(0, 1.1)
        # Fully open
        sim.partial_opening_policy(0, 0)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert not node.stay_at_home
            if node.old_active:
                assert node.active
                assert node.output == node.old_output
            else:
                assert not node.active
                assert node.output == 0
        # Fully closed but staying at home does not affect output
        sim.partial_opening_policy(1, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation == 'worker':
                assert node.stay_at_home
                if node.old_active:
                    assert node.active
                    assert node.output == node.old_output
                else:
                    assert not node.active
                    assert node.output == 0
        # Fully closed and staying at home halves the output
        output_remaining_scale = 0.5
        sim.partial_opening_policy(1, output_remaining_scale)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation == 'worker':
                assert node.stay_at_home
                if node.old_active:
                    assert node.active
                    assert node.output == node.old_output * output_remaining_scale
                else:
                    assert not node.active
                    assert node.output == 0

    def test_reopen_policy(self):
        sim = Simulation()
        sim.num_nodes = 6
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        for i in range(sim.num_nodes - 2):
            sim.node_dict[i].occupation = 'worker'
        sim.node_dict[4].occupation = 'student'
        sim.node_dict[5].occupation = 'retiree'
        for i in range(sim.num_nodes):
            sim.node_dict[i].stay_at_home = True
        sim.node_dict[1].isolated = True
        sim.node_dict[2].hospitalized = True
        sim.node_dict[3].deceased = True
        sim.reopen_policy()
        assert not sim.node_dict[0].stay_at_home
        for i in range(1, sim.num_nodes):
            assert sim.node_dict[i].stay_at_home

    def test_infect_node_no_vulnerable_group(self):
        sim = Simulation()
        sim.num_nodes = 20
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.vulnerable_population_fraction = 0
        with pytest.raises(AssertionError, match='Probability of transmission must be between 0 and 1'):
            sim.infect_node(0, -0.1)
        with pytest.raises(AssertionError, match='Probability of transmission must be between 0 and 1'):
            sim.infect_node(0, 1.1)
        sim.node_dict[0].infected = True
        assert not sim.infect_node(0, 1)
        sim.node_dict[1].deceased = True
        assert not sim.infect_node(1, 1)
        sim.node_dict[2].recovered = True
        assert not sim.infect_node(2, 1)
        assert not sim.infect_node(3, 0)
        for i in range(4, sim.num_nodes):
            node = sim.node_dict[i]
            res = sim.infect_node(i, 1)
            assert res
            assert node.days_infected == 1
            assert node.disease_length >= max(node.symptom_date, node.hospital_date, node.ICU_date, node.death_date)
            if node.symptom_date == 0:
                assert node.hospital_date == 0
                assert node.ICU_date == 0
                assert node.death_date == 0
            elif node.hospital_date == 0:
                assert node.ICU_date == 0
                assert node.death_date == 0
            elif node.ICU_date == 0:
                assert node.death_date == 0
            if node.death_date > 0:
                assert 0 < node.symptom_date <= node.hospital_date <= node.ICU_date <= node.death_date
            else:
                if node.ICU_date > 0:
                    assert 0 < node.symptom_date <= node.hospital_date <= node.ICU_date
                else:
                    if node.hospital_date > 0:
                        assert 0 < node.symptom_date <= node.hospital_date

    def test_infect_node_vulnerable_group(self):
        sim = Simulation()
        sim.num_nodes = 20
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.vulnerable_population_fraction = 0
        with pytest.raises(AssertionError, match='Probability of transmission must be between 0 and 1'):
            sim.infect_node(0, -0.1)
        with pytest.raises(AssertionError, match='Probability of transmission must be between 0 and 1'):
            sim.infect_node(0, 1.1)
        sim.node_dict[0].infected = True
        assert not sim.infect_node(0, 1)
        sim.node_dict[1].deceased = True
        assert not sim.infect_node(1, 1)
        sim.node_dict[2].recovered = True
        assert not sim.infect_node(2, 1)
        assert not sim.infect_node(3, 0)
        for i in range(4, sim.num_nodes):
            node = sim.node_dict[i]
            res = sim.infect_node(i, 1)
            assert res
            assert node.days_infected == 1
            assert node.disease_length >= max(node.symptom_date, node.hospital_date, node.ICU_date, node.death_date)
            if node.symptom_date == 0:
                assert node.hospital_date == 0
                assert node.ICU_date == 0
                assert node.death_date == 0
            elif node.hospital_date == 0:
                assert node.ICU_date == 0
                assert node.death_date == 0
            elif node.ICU_date == 0:
                assert node.death_date == 0
            if node.death_date > 0:
                assert 0 < node.symptom_date <= node.hospital_date <= node.ICU_date <= node.death_date
            else:
                if node.ICU_date > 0:
                    assert 0 < node.symptom_date <= node.hospital_date <= node.ICU_date
                else:
                    if node.hospital_date > 0:
                        assert 0 < node.symptom_date <= node.hospital_date

    def test_infect_node_compare_vulnerable_normal(self):
        sim = Simulation()
        sim.num_nodes = 20
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.vulnerable_population_fraction = .5
        sim.vulnerability_indices['vulnerable'] = 1e4
        sim.vulnerability_indices['normal'] = 0
        for i in range(int(sim.num_nodes * sim.vulnerable_population_fraction)):
            sim.node_dict[i].vulnerable = True
        sim.min_disease_length = sim.max_disease_length = 7
        sim.min_asymptomatic_length = sim.max_asymptomatic_length = 2
        sim.min_hospital_after_symptoms = sim.max_hospital_after_symptoms = 1
        sim.max_ICU_after_hospital = 0
        sim.max_death_after_ICU = 0
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            sim.infect_node(i, 1)
            if node.vulnerable:
                assert node.disease_length == (sim.min_disease_length + sim.min_asymptomatic_length + 1
                                               + sim.min_hospital_after_symptoms + sim.max_ICU_after_hospital
                                               + sim.max_death_after_ICU)
                assert node.symptom_date == sim.min_asymptomatic_length + 1
                assert node.hospital_date == (sim.min_asymptomatic_length + 1 + sim.min_hospital_after_symptoms)
                assert node.ICU_date == (sim.min_asymptomatic_length + 1 + sim.min_hospital_after_symptoms
                                         + sim.max_ICU_after_hospital)
                assert node.death_date == (sim.min_asymptomatic_length + 1 + sim.min_hospital_after_symptoms
                                           + sim.max_ICU_after_hospital + sim.max_death_after_ICU)
            else:
                assert node.disease_length == sim.min_disease_length
                assert node.symptom_date == 0
                assert node.hospital_date == 0
                assert node.ICU_date == 0
                assert node.death_date == 0

    def test_check_hospital_capacity_no_household_grouping_or_vulnerable_group(self):
        sim = Simulation()
        num_households = 10
        sim.household_network(num_households, 1, None, 1.0)
        with pytest.raises(AssertionError, match='Capacity must be positive and no more than 1'):
            sim.check_hospital_capacity(0, 0.2)
        with pytest.raises(AssertionError, match='Capacity must be positive and no more than 1'):
            sim.check_hospital_capacity(1.1, 0.2)
        with pytest.raises(AssertionError, match='The effect of undertreatment must be nonnegative'):
            sim.check_hospital_capacity(0.5, -0.1)
        hospital_capacity = .1
        undertreatment_effect = 1e5
        # No one infected
        sim.hospitalized_nodes = []
        sim.check_hospital_capacity(hospital_capacity, undertreatment_effect)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert not node.infected
            assert not node.hospitalized
            assert not node.undertreated
            assert not node.ICU
            assert not node.deceased
            assert node.disease_length == 0
            assert node.days_infected == 0
            assert node.symptom_date == 0
            assert node.hospital_date == 0
            assert node.ICU_date == 0
            assert node.death_date == 0
        # Hospital capacity exceeded and everyone hospitalized dies eventually
        for i in range(2):
            node = sim.node_dict[i]
            node.infected = True
            node.hospitalized = True
            sim.hospitalized_nodes.append(node)
            node.disease_length = 10
            node.symptom_date = 1
            node.hospital_date = 1
            node.undertreated = False
        sim.node_dict[0].days_infected = 2
        sim.node_dict[0].ICU_date = 2
        sim.node_dict[0].ICU = True
        sim.node_dict[1].days_infected = 1
        sim.node_dict[1].ICU_date = 9
        sim.node_dict[1].ICU = False
        sim.max_ICU_after_hospital = 1
        sim.max_death_after_ICU = 1
        sim.check_hospital_capacity(hospital_capacity, undertreatment_effect)
        for i in range(2):
            assert sim.node_dict[i].undertreated
            sim.node_dict[i].infected_update_day()
        if not sim.node_dict[1].deceased:
            sim.node_dict[1].infected_update_day()
        for i in range(2):
            node = sim.node_dict[i]
            assert node.undertreated
            assert node.deceased
            assert node.days_infected <= 3
            assert node.ICU_date <= 2
            assert node.death_date <= 3

    def test_check_hospital_capacity_household_grouping_no_vulnerable_group(self):
        sim = Simulation()
        num_households = 10
        sim.household_network((num_households // 2), 1, 'poor', 1.0)
        sim.household_network((num_households // 2), 1, 'rich', 1.0)
        hospital_capacity = .1
        undertreatment_effect = 1e5
        # Hospital capacity exceeded and everyone hospitalized who is also poor dies eventually
        for i in {0, 1, sim.num_nodes - 1}:
            node = sim.node_dict[i]
            node.infected = True
            node.hospitalized = True
            sim.hospitalized_nodes.append(node)
            node.disease_length = 10
            node.symptom_date = 1
            node.hospital_date = 1
            node.undertreated = False
        sim.node_dict[0].days_infected = 2
        sim.node_dict[0].ICU_date = 2
        sim.node_dict[0].ICU = True
        sim.node_dict[1].days_infected = 1
        sim.node_dict[1].ICU_date = 9
        sim.node_dict[1].ICU = False
        sim.node_dict[sim.num_nodes - 1].days_infected = 1
        sim.node_dict[sim.num_nodes - 1].ICU_date = 9
        sim.node_dict[sim.num_nodes - 1].ICU = False
        sim.max_ICU_after_hospital = 1
        sim.max_death_after_ICU = 1
        sim.check_hospital_capacity(hospital_capacity, undertreatment_effect)
        assert sim.node_dict[0].undertreated
        assert sim.node_dict[1].undertreated
        assert not sim.node_dict[sim.num_nodes - 1].undertreated
        for i in {0, 1, sim.num_nodes - 1}:
            sim.node_dict[i].infected_update_day()
        if not sim.node_dict[1].deceased:
            sim.node_dict[1].infected_update_day()
        for _ in range(sim.node_dict[sim.num_nodes - 1].ICU_date // 2):
            sim.node_dict[sim.num_nodes - 1].infected_update_day()
        for i in range(2):
            node = sim.node_dict[i]
            assert node.undertreated
            assert node.deceased == 'undertreatment'
            assert node.days_infected <= 3
            assert node.ICU_date <= 2
            assert node.death_date <= 3
        node = sim.node_dict[sim.num_nodes - 1]
        assert not node.undertreated
        assert not node.deceased
        assert node.days_infected == (2 + sim.node_dict[sim.num_nodes - 1].ICU_date // 2)
        assert node.ICU_date == 9
        assert node.death_date == 0

    def test_check_hospital_capacity_vulnerable_group(self):
        sim = Simulation()
        num_households = 10
        sim.household_network(num_households, 1, None, 1.0)
        hospital_capacity = .1
        undertreatment_effect = 0
        sim.vulnerable_population_fraction = .1
        sim.vulnerability_indices = {'vulnerable': 1e5, 'normal': 0}
        # Hospital capacity exceeded and everyone hospitalized that is vulnerable dies eventually
        for i in range(2):
            node = sim.node_dict[i]
            node.infected = True
            node.hospitalized = True
            sim.hospitalized_nodes.append(node)
            node.disease_length = 10
            node.symptom_date = 1
            node.hospital_date = 1
            node.undertreated = False
        sim.node_dict[0].vulnerable = False
        sim.node_dict[0].days_infected = 2
        sim.node_dict[0].ICU_date = 2
        sim.node_dict[0].ICU = True
        sim.node_dict[0].death_date = 9
        sim.node_dict[1].vulnerable = True
        sim.node_dict[1].days_infected = 1
        sim.node_dict[1].ICU_date = 9
        sim.node_dict[1].ICU = False
        sim.max_ICU_after_hospital = 1
        sim.max_death_after_ICU = 1
        sim.check_hospital_capacity(hospital_capacity, undertreatment_effect)
        for i in range(2):
            assert sim.node_dict[i].undertreated
            sim.node_dict[i].infected_update_day()
        node = sim.node_dict[0]
        assert not node.deceased
        assert node.days_infected == 3
        assert node.ICU_date == 2
        assert node.death_date == 9
        if not sim.node_dict[1].deceased:
            sim.node_dict[1].infected_update_day()
        node = sim.node_dict[1]
        assert node.undertreated
        assert node.deceased
        assert node.days_infected <= 3
        assert node.ICU_date <= 2
        assert node.death_date <= 3

    def test_calc_output_measures(self):
        sim = Simulation()
        sim.output_scale_half_workers = 0.25
        with pytest.raises(AssertionError, match='output_scale_half_workers must be between 0.25 and 1 exclusive'):
            sim.calc_output_measures(1)
        sim.output_scale_half_workers = 1
        with pytest.raises(AssertionError, match='output_scale_half_workers must be between 0.25 and 1 exclusive'):
            sim.calc_output_measures(1)
        sim.output_scale_half_workers = 0.25 + 0.75 * random.random()
        with pytest.raises(AssertionError, match='Average worker degree must be positive'):
            sim.calc_output_measures(0)
        average_worker_degree = random.randint(1, 50)
        sim.calc_output_measures(average_worker_degree)
        assert sim.personal_to_linkage_output_ratio == ((sim.output_scale_half_workers - 0.25)
                                                        / (1 - sim.output_scale_half_workers))
        assert sim.colleague_linkage_output == Node.personal_output / (sim.personal_to_linkage_output_ratio
                                                                           * average_worker_degree)

    def test_sigmoid(self):
        sim = Simulation()
        sim.inflection_x = 1
        with pytest.raises(AssertionError, match='Parameter nu must be positive'):
            sim.sigmoid(0, 0)
        for nu in {1e-4, 1e-3, 1e-2}:
            def first_derivative(x):
                power = nu ** (x / sim.inflection_x)
                res = math.log(nu) / (-nu * sim.inflection_x) * power * (1 + power) ** (-1 / nu - 1)
                return res

            def second_derivative(x):
                power = nu ** (x / sim.inflection_x)
                res = (math.log(nu) ** 2 / (-nu * sim.inflection_x ** 2) * power * (1 + power) ** (-1 / nu - 1)
                       * (1 - (1 / nu + 1) * power / (1 + power)))
                return res

            assert 0 <= sim.sigmoid(0, nu) < 1e-6
            assert (1 - 1e-6) < sim.sigmoid(5, nu) <= 1
            assert first_derivative(sim.inflection_x) >= first_derivative(sim.inflection_x - 1e-2)
            assert first_derivative(sim.inflection_x) >= first_derivative(sim.inflection_x + 1e-2)
            assert abs(second_derivative(sim.inflection_x)) < 1e-6

    def test_check_despair_no_one_home(self):
        sim = Simulation()
        sim.node_dict = {i: Node(sample_age()) for i in range(10)}
        with pytest.raises(AssertionError, match='The factor of probability of despair must be between 0 and 1'):
            sim.check_despair(-0.1, None)
        with pytest.raises(AssertionError, match='The factor of probability of despair must be between 0 and 1'):
            sim.check_despair(1.1, None)
        with pytest.raises(AssertionError, match='Subsidy must be None or a nonnegative number'):
            sim.check_despair(.5, True)
        with pytest.raises(AssertionError, match='Subsidy must be None or a nonnegative number'):
            sim.check_despair(.5, -0.1)
        sim.node_dict = {}
        num_households = 100
        sim.household_network(num_households)
        for i in range(sim.num_nodes):
            if random.random() < 0.5:
                sim.node_dict[i].hospitalized = True
            else:
                sim.node_dict[i].deceased = True
        sim.check_despair(1)
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].deceased != 'despair'

    def test_check_despair_no_subsidy(self):
        sim = Simulation()
        sim.num_nodes = 8
        average_worker_degree = 2
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.calc_output_measures(average_worker_degree)
        sim.household_grouping = False
        sim.household_dict[0] = {'members': {0, 1, 2}, 'type': None}
        sim.household_dict[1] = {'members': {3, 4, 5}, 'type': None}
        sim.household_dict[2] = {'members': {6}, 'type': None}
        sim.household_dict[3] = {'members': {7}, 'type': None}
        sim.node_dict[0].hospitalized = True
        sim.node_dict[1].deceased = True
        sim.node_dict[3].hospitalized = True
        sim.node_dict[4].deceased = True
        for i in {0, 1, 3, 4, 5, 7}:
            sim.node_dict[i].active = False
            sim.node_dict[i].output = 0
        sim.node_dict[2].active = True
        sim.node_dict[2].output = 10
        sim.node_dict[2].colleagues = {6}
        sim.node_dict[6].active = True
        sim.node_dict[6].output = 1
        sim.node_dict[6].colleagues = {2, 7}
        sim.node_dict[7].active = True
        sim.node_dict[7].colleagues = {6}
        total_subsidy = sim.check_despair(1, 0)
        assert total_subsidy == 0
        for i in {0, 1, 2, 3, 4, 6}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[0].hospitalized = False
        sim.node_dict[0].isolated = True
        sim.node_dict[0].active = True
        sim.node_dict[0].output = 10
        sim.node_dict[0].colleagues = {6}
        sim.node_dict[2].stay_at_home = True
        sim.node_dict[6].colleagues.add(0)
        total_subsidy = sim.check_despair(1, 0)
        assert total_subsidy == 0
        for i in {0, 1, 2, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[2].output = 0
        total_subsidy = sim.check_despair(1, 0)
        assert total_subsidy == 0
        for i in {0, 1, 2, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[0].output = 1.5
        total_subsidy = sim.check_despair(1, 0)
        assert total_subsidy == 0
        for i in {1, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {0, 2, 5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'

    def test_check_despair_subsidy_as_needed(self):
        sim = Simulation()
        sim.num_nodes = 8
        average_worker_degree = 2
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.calc_output_measures(average_worker_degree)
        sim.household_grouping = False
        sim.household_dict[0] = {'members': {0, 1, 2}, 'type': None}
        sim.household_dict[1] = {'members': {3, 4, 5}, 'type': None}
        sim.household_dict[2] = {'members': {6}, 'type': None}
        sim.household_dict[3] = {'members': {7}, 'type': None}
        sim.node_dict[0].hospitalized = True
        sim.node_dict[1].deceased = True
        sim.node_dict[3].hospitalized = True
        sim.node_dict[4].deceased = True
        for i in {0, 1, 3, 4, 5, 7}:
            sim.node_dict[i].output = 0
        sim.node_dict[2].active = True
        sim.node_dict[2].output = 10
        sim.node_dict[6].active = True
        sim.node_dict[6].output = 1
        sim.node_dict[6].colleagues = {2, 7}
        sim.node_dict[7].active = True
        sim.check_despair(1, None)
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].deceased != 'despair'
        sim.node_dict[0].hospitalized = False
        sim.node_dict[0].isolated = True
        sim.node_dict[0].active = True
        sim.node_dict[0].output = 10
        sim.node_dict[2].stay_at_home = True
        sim.node_dict[6].colleagues = {0, 2}
        sim.check_despair(1, None)
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].deceased != 'despair'
        sim.node_dict[2].output = 0
        sim.check_despair(1, None)
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].deceased != 'despair'
        sim.node_dict[0].output = 2
        sim.check_despair(1, None)
        for i in range(sim.num_nodes):
            assert sim.node_dict[i].deceased != 'despair'

    def test_check_despair_fixed_subsidy_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 8
        average_worker_degree = 2
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.calc_output_measures(average_worker_degree)
        sim.household_grouping = True
        sim.household_dict[0] = {'members': {0, 1, 2}, 'type': 'rich'}
        sim.household_dict[1] = {'members': {3, 4, 5}, 'type': 'poor'}
        sim.household_dict[2] = {'members': {6}, 'type': 'poor'}
        sim.household_dict[3] = {'members': {7}, 'type': 'poor'}
        sim.node_dict[0].hospitalized = True
        sim.node_dict[1].deceased = True
        sim.node_dict[3].hospitalized = True
        sim.node_dict[4].deceased = True
        for i in {0, 1, 3, 4, 5, 7}:
            sim.node_dict[i].output = 0
        sim.node_dict[2].active = True
        sim.node_dict[2].output = 10
        sim.node_dict[6].active = True
        sim.node_dict[6].output = 1
        sim.node_dict[6].colleagues = {2, 7}
        sim.node_dict[7].active = True
        subsidy = 1e-4
        total_subsidy = sim.check_despair(1, subsidy)
        assert abs(total_subsidy['poor'] - subsidy * 2) < 1e-8
        assert total_subsidy['rich'] == 0
        for i in {0, 1, 2, 3, 4, 6}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[0].hospitalized = False
        sim.node_dict[0].isolated = True
        sim.node_dict[0].active = True
        sim.node_dict[0].output = 10
        sim.node_dict[2].stay_at_home = True
        sim.node_dict[6].colleagues = {0, 2}
        total_subsidy = sim.check_despair(1, subsidy)
        assert abs(total_subsidy['poor'] - subsidy) < 1e-8
        assert total_subsidy['rich'] == 0
        for i in {0, 1, 2, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[2].output = 0
        total_subsidy = sim.check_despair(1, subsidy)
        assert total_subsidy['poor'] == 0
        assert abs(total_subsidy['rich'] - subsidy) < 1e-8
        for i in {0, 1, 2, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'
        sim.node_dict[0].output = 2
        total_subsidy = sim.check_despair(1, subsidy)
        assert total_subsidy['poor'] == 0
        assert abs(total_subsidy['rich'] - subsidy) < 1e-8
        for i in {1, 3, 4}:
            assert sim.node_dict[i].deceased != 'despair'
        for i in {0, 2, 5, 6, 7}:
            assert sim.node_dict[i].deceased == 'despair'

    def test_check_output_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = False
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.initial_total_output = 0
        with pytest.raises(AssertionError, match='Total subsidy must be nonnegative'):
            sim.check_output(-0.1, .5)
        with pytest.raises(AssertionError, match='factor of probability of becoming inactive must be between 0 and 1'):
            sim.check_output(0, -0.1)
        with pytest.raises(AssertionError, match='factor of probability of becoming inactive must be between 0 and 1'):
            sim.check_output(0, 1.1)
        with pytest.raises(AssertionError, match='Initial total output must be positive'):
            sim.check_output(0, .5)
        # No loss in total output and hence no layoff
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            node.active = True
            node.output = 1
            node.old_active = True
            node.old_output = 1
            node.occupation = 'worker'
        sim.node_dict[0].occupation = 'student'
        sim.node_dict[1].occupation = 'retiree'
        sim.node_dict[2].hospitalized = True
        sim.node_dict[3].deceased = True
        sim.initial_total_output = sim.get_total_output()
        sim.check_output(0, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # Sufficient subsidy and hence no layoff
        sim.initial_total_output = 1e8
        sim.check_output(1e8, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # No effect of output loss on active rate and hence no layoff
        sim.initial_total_output = 1e8
        sim.check_output(0, 0)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # Every worker that is alive and not hospitalized gets laid off
        sim.check_output(0, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation == 'worker' and not node.hospitalized and not node.deceased:
                assert not node.active
                assert node.output == 0
                assert not node.old_active
                assert node.old_output == 0
            else:
                assert node.active
                assert node.output == 1
                assert node.old_active
                assert node.old_output == 1

    def test_check_output_household_grouping(self):
        sim = Simulation()
        num_households = 10
        sim.household_network((num_households // 2), 1, 'poor', 1.0)
        sim.household_network((num_households // 2), 1, 'rich', 1.0)
        sim.initial_total_output = {'poor': 0, 'rich': 0}
        with pytest.raises(AssertionError, match='Total subsidy must be either a nonnegative number or a dictionary'):
            sim.check_output([0, 0], .5)
        with pytest.raises(AssertionError, match='Total subsidy must be either a nonnegative number or a dictionary'):
            sim.check_output((0, 0), .5)
        with pytest.raises(AssertionError, match='Total subsidy must be either a nonnegative number or a dictionary'):
            sim.check_output(True, .5)
        with pytest.raises(AssertionError, match='Total subsidy must be nonnegative for every household group'):
            sim.check_output({'poor': -0.1, 'rich': 0}, .5)
        with pytest.raises(AssertionError, match='Initial total output must be positive'):
            sim.check_output({'poor': 0, 'rich': 0}, .5)
        # No loss in total output and hence no layoff
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            node.active = True
            node.output = 1
            node.old_active = True
            node.old_output = 1
            node.occupation = 'worker'
        sim.node_dict[0].occupation = 'student'
        sim.node_dict[1].occupation = 'retiree'
        sim.node_dict[2].hospitalized = True
        sim.node_dict[3].deceased = True
        sim.initial_total_output = sim.get_total_output()
        sim.check_output({'poor': 0, 'rich': 0}, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # Sufficient subsidy and hence no layoff
        sim.initial_total_output = {'poor': 1e8, 'rich': 1e8}
        sim.check_output({'poor': 1e8, 'rich': 1e8}, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # No effect of output loss on active rate and hence no layoff
        sim.initial_total_output = {'poor': 1e8, 'rich': 1e8}
        sim.check_output({'poor': 0, 'rich': 0}, 0)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            assert node.active
            assert node.output == 1
            assert node.old_active
            assert node.old_output == 1
        # Every worker that is alive and not hospitalized gets laid off
        sim.check_output({'poor': 0, 'rich': 0}, 1)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            if node.occupation == 'worker' and not node.hospitalized and not node.deceased:
                assert not node.active
                assert node.output == 0
                assert not node.old_active
                assert node.old_output == 0
            else:
                assert node.active
                assert node.output == 1
                assert node.old_active
                assert node.old_output == 1

    def test_contact_tracing_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = False
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': None, 'members': {0, 1, 2}},
                              1: {'type': None, 'members': {3}},
                              2: {'type': None, 'members': {4}},
                              3: {'type': None, 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        for i in {0, 1, 3, 7, 8}:
            sim.node_dict[i].active = True
        for i in {2, 4, 5, 6, 9}:
            sim.node_dict[i].active = False
        sim.node_dict[0].connections = [1]
        sim.node_dict[1].connections = [0, 3, 5, 6, 7]
        sim.node_dict[3].connections = [1, 7, 8]
        sim.node_dict[5].connections = [1]
        sim.node_dict[6].connections = [1]
        sim.node_dict[7].connections = [1, 3]
        sim.node_dict[8].connections = [3]
        sim.node_dict[2].isolated = True
        sim.node_dict[3].stay_at_home = True
        sim.node_dict[5].hospitalized = True
        sim.node_dict[6].deceased = True
        with pytest.raises(AssertionError, match='Efficacy must be between 0 and 1'):
            sim.contact_tracing(0, -0.1)
        with pytest.raises(AssertionError, match='Efficacy must be between 0 and 1'):
            sim.contact_tracing(0, 1.1)
        # Contact tracing completely ineffective for non-household connections
        contacts = sim.contact_tracing(0, 0)
        assert contacts == {1}
        contacts = sim.contact_tracing(1, 0)
        assert contacts == {0}
        contacts = sim.contact_tracing(2, 0)
        assert contacts == {0, 1}
        contacts = sim.contact_tracing(3, 0)
        assert contacts == set()
        contacts = sim.contact_tracing(4, 0)
        assert contacts == set()
        contacts = sim.contact_tracing(5, 0)
        assert contacts == {7, 8, 9}
        contacts = sim.contact_tracing(6, 0)
        assert contacts == {7, 8, 9}
        contacts = sim.contact_tracing(7, 0)
        assert contacts == {8, 9}
        contacts = sim.contact_tracing(8, 0)
        assert contacts == {7, 9}
        contacts = sim.contact_tracing(9, 0)
        assert contacts == {7, 8}
        # Contact tracing completely effective
        contacts = sim.contact_tracing(0, 1)
        assert contacts == {1}
        contacts = sim.contact_tracing(1, 1)
        assert contacts == {0, 7}
        contacts = sim.contact_tracing(2, 1)
        assert contacts == {0, 1}
        contacts = sim.contact_tracing(3, 1)
        assert contacts == set()
        contacts = sim.contact_tracing(4, 1)
        assert contacts == set()
        contacts = sim.contact_tracing(5, 1)
        assert contacts == {7, 8, 9}
        contacts = sim.contact_tracing(6, 1)
        assert contacts == {7, 8, 9}
        contacts = sim.contact_tracing(7, 1)
        assert contacts == {1, 8, 9}
        contacts = sim.contact_tracing(8, 1)
        assert contacts == {7, 9}
        contacts = sim.contact_tracing(9, 1)
        assert contacts == {7, 8}

    def test_contact_tracing_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = True
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': 'rich', 'members': {0, 1, 2}},
                              1: {'type': 'rich', 'members': {3}},
                              2: {'type': 'rich', 'members': {4}},
                              3: {'type': 'poor', 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        for i in {0, 1, 3, 7, 8}:
            sim.node_dict[i].active = True
        for i in {2, 4, 5, 6, 9}:
            sim.node_dict[i].active = False
        sim.node_dict[0].connections = [1]
        sim.node_dict[1].connections = [0, 3, 5, 6, 7]
        sim.node_dict[3].connections = [1, 7, 8]
        sim.node_dict[5].connections = [1]
        sim.node_dict[6].connections = [1]
        sim.node_dict[7].connections = [1, 3]
        sim.node_dict[8].connections = [3]
        sim.node_dict[2].isolated = True
        sim.node_dict[3].stay_at_home = True
        sim.node_dict[5].hospitalized = True
        sim.node_dict[6].deceased = True
        with pytest.raises(AssertionError, match='Efficacy must be between 0 and 1'):
            sim.contact_tracing(0, -0.1)
        with pytest.raises(AssertionError, match='Efficacy must be between 0 and 1'):
            sim.contact_tracing(0, 1.1)
        # Contact tracing completely ineffective for non-household connections
        contacts = sim.contact_tracing(0, 0)
        assert contacts == {'poor': set(), 'rich': {1}}
        contacts = sim.contact_tracing(1, 0)
        assert contacts == {'poor': set(), 'rich': {0}}
        contacts = sim.contact_tracing(2, 0)
        assert contacts == {'poor': set(), 'rich': {0, 1}}
        contacts = sim.contact_tracing(3, 0)
        assert contacts == {'poor': set(), 'rich': set()}
        contacts = sim.contact_tracing(4, 0)
        assert contacts == {'poor': set(), 'rich': set()}
        contacts = sim.contact_tracing(5, 0)
        assert contacts == {'poor': {7, 8, 9}, 'rich': set()}
        contacts = sim.contact_tracing(6, 0)
        assert contacts == {'poor': {7, 8, 9}, 'rich': set()}
        contacts = sim.contact_tracing(7, 0)
        assert contacts == {'poor': {8, 9}, 'rich': set()}
        contacts = sim.contact_tracing(8, 0)
        assert contacts == {'poor': {7, 9}, 'rich': set()}
        contacts = sim.contact_tracing(9, 0)
        assert contacts == {'poor': {7, 8}, 'rich': set()}
        # Contact tracing completely effective
        contacts = sim.contact_tracing(0, 1)
        assert contacts == {'poor': set(), 'rich': {1}}
        contacts = sim.contact_tracing(1, 1)
        assert contacts == {'poor': {7}, 'rich': {0}}
        contacts = sim.contact_tracing(2, 1)
        assert contacts == {'poor': set(), 'rich': {0, 1}}
        contacts = sim.contact_tracing(3, 1)
        assert contacts == {'poor': set(), 'rich': set()}
        contacts = sim.contact_tracing(4, 1)
        assert contacts == {'poor': set(), 'rich': set()}
        contacts = sim.contact_tracing(5, 1)
        assert contacts == {'poor': {7, 8, 9}, 'rich': set()}
        contacts = sim.contact_tracing(6, 1)
        assert contacts == {'poor': {7, 8, 9}, 'rich': set()}
        contacts = sim.contact_tracing(7, 1)
        assert contacts == {'poor': {8, 9}, 'rich': {1}}
        contacts = sim.contact_tracing(8, 1)
        assert contacts == {'poor': {7, 9}, 'rich': set()}
        contacts = sim.contact_tracing(9, 1)
        assert contacts == {'poor': {7, 8}, 'rich': set()}

    def test_get_test_candidates_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 5
        sim.household_grouping = False
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.node_dict[0].isolated = True
        sim.node_dict[1].hospitalized = True
        sim.node_dict[2].deceased = True
        candidates = sim.get_test_candidates()
        assert candidates == {3, 4}

    def test_get_test_candidates_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = True
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': 'rich', 'members': {0, 1, 2}},
                              1: {'type': 'rich', 'members': {3}},
                              2: {'type': 'rich', 'members': {4}},
                              3: {'type': 'poor', 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        sim.node_dict[0].isolated = True
        sim.node_dict[3].hospitalized = True
        sim.node_dict[5].deceased = True
        candidates = sim.get_test_candidates()
        assert candidates == {'poor': {6, 7, 8, 9}, 'rich': {1, 2, 4}}

    def test_viral_test_no_household_grouping_no_contact_tracing(self):
        sim = Simulation()
        num_households = 10
        sim.household_grouping = False
        sim.household_network(num_households, 1, None, 1.0)
        with pytest.raises(AssertionError, match='Nodes to test must be a list, tuple, or numpy array'):
            sim.viral_test({0, 1, 2}, .9, .7)
        with pytest.raises(AssertionError, match='Nodes to test cannot be empty'):
            sim.viral_test([], .9, .7)
        with pytest.raises(AssertionError, match='Nodes to test cannot be empty'):
            sim.viral_test((), .9, .7)
        with pytest.raises(AssertionError, match='Nodes to test cannot be empty'):
            sim.viral_test(np.array([]), .9, .7)
        with pytest.raises(AssertionError, match='Sensitivity must be between 0 and 1'):
            sim.viral_test([0], -0.1, .7)
        with pytest.raises(AssertionError, match='Sensitivity must be between 0 and 1'):
            sim.viral_test([0], 1.1, .7)
        for i in range(sim.num_nodes // 2):
            sim.node_dict[i].infected = True
        # Zero testing sensitivity
        nodes_to_test = np.random.permutation(sim.num_nodes)
        sim.viral_test(nodes_to_test, 0, 0)
        assert sim.test_backlog == []
        for i in range(sim.num_nodes):
            assert not sim.node_dict[i].isolated
        # Perfect testing sensitivity
        nodes_to_test = np.random.permutation(sim.num_nodes)
        sim.viral_test(nodes_to_test, 1, 0)
        assert sim.test_backlog == []
        for i in range(sim.num_nodes // 2):
            assert sim.node_dict[i].isolated
        for i in range((sim.num_nodes // 2), sim.num_nodes):
            assert not sim.node_dict[i].isolated

    def test_viral_test_no_household_grouping_with_contact_tracing(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = False
        sim.test_backlog = []
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': None, 'members': {0, 1, 2}},
                              1: {'type': None, 'members': {3}},
                              2: {'type': None, 'members': {4}},
                              3: {'type': None, 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        for i in {0, 1, 3, 4, 7, 8}:
            sim.node_dict[i].active = True
        for i in {2, 5, 6, 9}:
            sim.node_dict[i].active = False
        sim.node_dict[0].connections = [1, 4]
        sim.node_dict[1].connections = [0, 3, 5, 6, 7]
        sim.node_dict[3].connections = [1, 7, 8]
        sim.node_dict[4].connections = [0]
        sim.node_dict[5].connections = [1]
        sim.node_dict[6].connections = [1]
        sim.node_dict[7].connections = [1, 3]
        sim.node_dict[8].connections = [3]
        sim.node_dict[2].stay_at_home = True
        sim.node_dict[5].hospitalized = True
        sim.node_dict[6].deceased = True
        for i in {0, 4, 5}:
            sim.node_dict[i].infected = True
        # Perfect testing sensitivity and perfect contact tracing
        sim.viral_test([0, 4], 1, 1)
        for i in range(sim.num_nodes):
            if i in {0, 4}:
                assert sim.node_dict[i].isolated
            else:
                assert not sim.node_dict[i].isolated
        sim.test_backlog.sort()
        assert sim.test_backlog == [1, 2]
        sim.viral_test([5], 1, 1)
        for i in range(sim.num_nodes):
            if i in {0, 4, 5}:
                assert sim.node_dict[i].isolated
            else:
                assert not sim.node_dict[i].isolated
        sim.test_backlog.sort()
        assert sim.test_backlog == [1, 2, 7, 8, 9]

    def test_viral_test_household_grouping_no_contact_tracing(self):
        sim = Simulation()
        num_households = 10
        sim.household_grouping = True
        sim.household_network((num_households // 2), 1, 'poor', 1.0)
        sim.household_network((num_households // 2), 1, 'rich', 1.0)
        for i in range(sim.num_nodes // 2):
            sim.node_dict[i].infected = True
        # Zero testing sensitivity
        nodes_to_test = np.random.permutation(sim.num_nodes)
        sim.viral_test(nodes_to_test, 0, 0)
        assert sim.test_backlog == {'poor': [], 'rich': []}
        for i in range(sim.num_nodes):
            assert not sim.node_dict[i].isolated
        # Perfect testing sensitivity
        nodes_to_test = np.random.permutation(sim.num_nodes)
        sim.viral_test(nodes_to_test, 1, 0)
        assert sim.test_backlog == {'poor': [], 'rich': []}
        for i in range(sim.num_nodes // 2):
            assert sim.node_dict[i].isolated
        for i in range((sim.num_nodes // 2), sim.num_nodes):
            assert not sim.node_dict[i].isolated

    def test_viral_test_household_grouping_and_contact_tracing(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.household_grouping = True
        sim.test_backlog = {'poor': [], 'rich': []}
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': 'rich', 'members': {0, 1, 2}},
                              1: {'type': 'rich', 'members': {3}},
                              2: {'type': 'rich', 'members': {4}},
                              3: {'type': 'poor', 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        for i in {0, 1, 3, 4, 7, 8}:
            sim.node_dict[i].active = True
        for i in {2, 5, 6, 9}:
            sim.node_dict[i].active = False
        sim.node_dict[0].connections = [1, 4]
        sim.node_dict[1].connections = [0, 3, 5, 6, 7]
        sim.node_dict[3].connections = [1, 7, 8]
        sim.node_dict[4].connections = [0]
        sim.node_dict[5].connections = [1]
        sim.node_dict[6].connections = [1]
        sim.node_dict[7].connections = [1, 3]
        sim.node_dict[8].connections = [3]
        sim.node_dict[2].stay_at_home = True
        sim.node_dict[5].hospitalized = True
        sim.node_dict[6].deceased = True
        for i in {0, 4, 5}:
            sim.node_dict[i].infected = True
        # Perfect testing sensitivity and perfect contact tracing
        sim.viral_test([0, 4], 1, 1)
        for i in range(sim.num_nodes):
            if i in {0, 4}:
                assert sim.node_dict[i].isolated
            else:
                assert not sim.node_dict[i].isolated
        sim.test_backlog['rich'].sort()
        assert sim.test_backlog['poor'] == []
        assert sim.test_backlog['rich'] == [1, 2]
        sim.viral_test([5], 1, 1)
        for i in range(sim.num_nodes):
            if i in {0, 4, 5}:
                assert sim.node_dict[i].isolated
            else:
                assert not sim.node_dict[i].isolated
        sim.test_backlog['poor'].sort()
        sim.test_backlog['rich'].sort()
        assert sim.test_backlog['poor'] == [7, 8, 9]
        assert sim.test_backlog['rich'] == [1, 2]

    def test_random_testing_protocol_check_assertions(self):
        sim = Simulation()
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol(-0.1, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol(1.0, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol(1.1, .9, .7)
        sim.household_grouping = False
        with pytest.raises(AssertionError, match='Household grouping must be used for differential testing'):
            sim.random_testing_protocol({}, .9, .7)
        sim.household_grouping = True
        with pytest.raises(AssertionError, match='Number of tests must have poor as a key'):
            sim.random_testing_protocol({}, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must have rich as a key'):
            sim.random_testing_protocol({'poor': 1}, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol({'poor': 1, 'rich': 1.0}, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol({'poor': 1.1, 'rich': 1}, .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol((1, 1), .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol([1, 1], .9, .7)
        with pytest.raises(AssertionError, match='Number of tests must be either a nonnegative integer or a dict'):
            sim.random_testing_protocol({'poor': -0.1, 'rich': 0}, .9, .7)

    def test_random_testing_protocol_no_household_grouping(self, capsys):
        sim = Simulation()
        num_households = 7
        sim.household_grouping = False
        sim.household_network(num_households, 1, None, 1.0)
        # No candidates left for testing
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = True
        sim.random_testing_protocol(1, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == '[Day 0] Warning: No candidates left for testing.\n'
        assert captured.err == ''
        assert sim.test_backlog == []
        # Not enough candidates left for testing
        sim.node_dict[0].isolated = False
        sim.node_dict[0].infected = True
        sim.test_backlog = [0]
        sim.random_testing_protocol(sim.num_nodes, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == f'[Day 0] Warning: Fewer than {sim.num_nodes} candidates left. Everyone testable is ' \
                               'tested.\n'
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert sim.test_backlog == []
        # Enough candidates for testing and the number of tests is less than the backlog
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = False
        sim.node_dict[0].connections = [1, 2, 3]
        sim.node_dict[1].connections = [0]
        sim.node_dict[2].connections = [0]
        sim.node_dict[3].connections = [0, 4]
        sim.node_dict[4].connections = [3]
        sim.node_dict[0].infected = True
        sim.node_dict[4].isolated = True
        sim.test_backlog = [0, 4]
        sim.random_testing_protocol(1, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert sim.test_backlog == [1, 2, 3]
        # Enough candidates for testing and the number of tests is more than the backlog
        for i in [1, 2, 3, 5, 6]:
            sim.node_dict[i].infected = True
        sim.random_testing_protocol(4, 1, 1)
        for i in [1, 2, 3]:
            assert sim.node_dict[i].isolated
        if sim.node_dict[5].isolated:
            assert not sim.node_dict[6].isolated
        else:
            assert not sim.node_dict[5].isolated
        assert sim.test_backlog == []

    def test_random_testing_protocol_household_grouping_uniform_testing(self, capsys):
        sim = Simulation()
        sim.household_network(5, 1, 'poor', 1.0)
        sim.household_network(2, 1, 'rich', 1.0)
        # No candidates left for testing
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = True
        sim.random_testing_protocol(1, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == '[Day 0] Warning: No candidates left for testing.\n'
        assert captured.err == ''
        assert sim.test_backlog == {'poor': [], 'rich': []}
        # Not enough candidates left for testing
        sim.node_dict[0].isolated = False
        sim.node_dict[0].infected = True
        sim.test_backlog = {'poor': [0], 'rich': []}
        sim.random_testing_protocol(sim.num_nodes, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == f'[Day 0] Warning: Fewer than {sim.num_nodes} candidates left. Everyone testable is ' \
                               'tested.\n'
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert sim.test_backlog == {'poor': [], 'rich': []}
        # Enough candidates for testing and the number of tests is less than the backlog
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = False
        sim.node_dict[0].connections = [1, 2, 3]
        sim.node_dict[1].connections = [0]
        sim.node_dict[2].connections = [0]
        sim.node_dict[3].connections = [0, 6]
        sim.node_dict[6].connections = [3]
        sim.node_dict[0].infected = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [0], 'rich': [6]}
        sim.random_testing_protocol(1, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert sim.test_backlog == {'poor': [1, 2, 3], 'rich': []}
        # Enough candidates for testing and the number of tests is more than the backlog
        for i in [1, 2, 3, 5, 6]:
            sim.node_dict[i].infected = True
        sim.random_testing_protocol(4, 1, 1)
        for i in [1, 2, 3]:
            assert sim.node_dict[i].isolated
        if sim.node_dict[4].isolated:
            assert not sim.node_dict[5].isolated
        else:
            assert not sim.node_dict[4].isolated
        assert sim.test_backlog == {'poor': [], 'rich': []}

    def test_random_testing_protocol_household_grouping_differential_testing(self, capsys):
        sim = Simulation()
        sim.household_network(5, 1, 'poor', 1.0)
        sim.household_network(3, 1, 'rich', 1.0)
        # No candidates left for testing
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = True
        sim.random_testing_protocol({'poor': 1, 'rich': 1}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == '[Day 0] Warning: No candidates left for testing.\n'
        assert captured.err == ''
        assert sim.test_backlog == {'poor': [], 'rich': []}
        # Not enough candidates left for testing
        sim.node_dict[0].isolated = False
        sim.node_dict[0].infected = True
        sim.test_backlog = {'poor': [0], 'rich': []}
        sim.random_testing_protocol({'poor': sim.num_nodes_poor, 'rich': sim.num_nodes_rich}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == f'[Day 0] Warning: Fewer than {sim.num_nodes} candidates left. Everyone testable is ' \
                               'tested.\n'
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert sim.test_backlog == {'poor': [], 'rich': []}
        # Enough candidates for testing in both groups and the number of tests is less than the backlog for the rich
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = False
        sim.node_dict[0].connections = [1, 2, 3]
        sim.node_dict[1].connections = [0]
        sim.node_dict[2].connections = [0]
        sim.node_dict[3].connections = [0, 6]
        sim.node_dict[6].connections = [3, 7]
        sim.node_dict[7].connections = [6]
        for i in [0, 1, 2, 3, 4, 5, 7]:
            sim.node_dict[i].infected = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [0, 4], 'rich': [6, 7, 5]}
        sim.random_testing_protocol({'poor': 3, 'rich': 1}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        for i in [0, 4, 7]:
            assert sim.node_dict[i].isolated
        assert not sim.node_dict[5].isolated
        if sim.node_dict[1].isolated:
            assert not sim.node_dict[2].isolated
            assert not sim.node_dict[3].isolated
            assert sim.test_backlog == {'poor': [2, 3], 'rich': [5]}
        elif sim.node_dict[2].isolated:
            assert not sim.node_dict[1].isolated
            assert not sim.node_dict[3].isolated
            assert sim.test_backlog == {'poor': [1, 3], 'rich': [5]}
        else:
            assert sim.node_dict[3].isolated
            assert not sim.node_dict[1].isolated
            assert not sim.node_dict[2].isolated
            assert sim.test_backlog == {'poor': [1, 2], 'rich': [5]}
        # Enough candidates for testing in both groups and the number of tests is less than the backlog for the poor
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = False
        for i in [0, 4, 5, 7]:
            sim.node_dict[i].infected = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [0, 4], 'rich': [6]}
        sim.random_testing_protocol({'poor': 1, 'rich': 1}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        assert sim.node_dict[0].isolated
        assert not sim.node_dict[4].isolated
        if sim.node_dict[5].isolated:
            assert not sim.node_dict[7].isolated
        else:
            assert sim.node_dict[7].isolated
        assert sim.test_backlog == {'poor': [4, 1, 2, 3], 'rich': []}
        # Enough rich candidates for testing but not enough poor candidates
        sim.node_dict[sim.num_nodes] = Node(sample_age())
        sim.household_dict[sim.num_nodes] = {'type': 'rich', 'members': {sim.num_nodes}}
        sim.node_dict[sim.num_nodes].household = sim.num_nodes
        sim.num_nodes += 1
        sim.num_nodes_rich += 1
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = False
        for i in [0, 4, 5, 6, 7, 8]:
            sim.node_dict[i].infected = True
        sim.node_dict[0].isolated = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [0], 'rich': [5]}
        sim.random_testing_protocol({'poor': sim.num_nodes_poor, 'rich': 1}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        for i in [0, 4, 5, 6]:
            assert sim.node_dict[i].isolated
        for i in [1, 2, 3]:
            assert not sim.node_dict[i].isolated
        if sim.node_dict[7].isolated:
            assert not sim.node_dict[8].isolated
        else:
            assert sim.node_dict[8].isolated
        assert sim.test_backlog == {'poor': [], 'rich': []}
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            if i in [0, 4, 5, 6, 7, 8]:
                sim.node_dict[i].infected = True
            else:
                sim.node_dict[i].infected = False
        sim.node_dict[0].isolated = True
        sim.node_dict[4].isolated = True
        sim.node_dict[5].connections = [7, 8]
        sim.node_dict[7].connections.append(5)
        sim.node_dict[8].connections = [5]
        sim.test_backlog = {'poor': [1], 'rich': [5, 8, 7, 6]}
        sim.random_testing_protocol({'poor': 4, 'rich': 1}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        for i in [0, 4, 5, 8]:
            assert sim.node_dict[i].isolated
        for i in [1, 2, 3, 6, 7]:
            assert not sim.node_dict[i].isolated
        assert sim.test_backlog == {'poor': [], 'rich': [7, 6]}
        # Enough poor candidates for testing but not enough rich candidates
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = True
        sim.node_dict[0].isolated = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [0], 'rich': [5]}
        sim.random_testing_protocol({'poor': 1, 'rich': sim.num_nodes_rich}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        for i in [0, 5, 6, 7, 8]:
            assert sim.node_dict[i].isolated
        count = 0
        for i in range(1, 5):
            if sim.node_dict[i].isolated:
                count += 1
        assert count == 2
        assert sim.test_backlog == {'poor': [], 'rich': []}
        for i in range(sim.num_nodes):
            sim.node_dict[i].isolated = False
            sim.node_dict[i].infected = True
        sim.node_dict[0].isolated = True
        sim.node_dict[6].isolated = True
        sim.test_backlog = {'poor': [3, 1, 2], 'rich': [5]}
        sim.random_testing_protocol({'poor': 1, 'rich': sim.num_nodes_rich}, 1, 1)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        for i in [0, 1, 3, 5, 6, 7, 8]:
            assert sim.node_dict[i].isolated
        for i in [2, 4]:
            assert not sim.node_dict[i].isolated
        assert sim.test_backlog == {'poor': [2], 'rich': []}

    def test_seed_simulation_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 10
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_grouping = False
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation([], record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation({}, record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation(None, record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation([0, 0, 1], record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation((0, 0, 1), record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation(np.array([0, 0, 1]), record_stats=False)
        random_output = random.random()
        for i in range(sim.num_nodes):
            sim.node_dict[i].active = True
            sim.node_dict[i].output = random_output
        infected = np.random.choice(range(sim.num_nodes), size=(sim.num_nodes // 2), replace=False)
        sim.seed_simulation(infected, record_stats=False)
        for i in range(sim.num_nodes):
            if i in infected:
                assert sim.node_dict[i].infected
            else:
                assert not sim.node_dict[i].infected
        assert abs(sim.initial_total_output - sim.num_nodes * random_output) < 1e-6

    def test_seed_simulation_household_grouping(self):
        sim = Simulation()
        num_households = 10
        output_poor = random.random()
        output_rich = random.random() * 10
        sim.household_network((num_households // 2), 1, 'poor', output_poor)
        sim.household_network((num_households // 2), 1, 'rich', output_rich)
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation([], record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation({}, record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections cannot be empty'):
            sim.seed_simulation(None, record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation([0, 0, 1], record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation((0, 0, 1), record_stats=False)
        with pytest.raises(AssertionError, match='Initial infections must be unique nodes'):
            sim.seed_simulation(np.array([0, 0, 1]), record_stats=False)
        for i in range(sim.num_nodes):
            sim.node_dict[i].active = True
        infected = np.random.choice(range(sim.num_nodes), size=(sim.num_nodes // 2), replace=False)
        sim.seed_simulation(infected, record_stats=False)
        for i in range(sim.num_nodes):
            if i in infected:
                assert sim.node_dict[i].infected
            else:
                assert not sim.node_dict[i].infected
        assert sim.initial_total_output['poor'] == (num_households // 2 * output_poor)
        assert sim.initial_total_output['rich'] == (num_households // 2 * output_rich)

    def test_simulation_step_no_household_grouping(self):
        time_steps = 10
        average_worker_degree = 2
        sim = Simulation()
        sim.calc_output_measures(average_worker_degree)
        sim.household_network(4, 3, None, 1.0)
        sim.econ_network(watts_strogatz_graph, [None, average_worker_degree, 0.5])
        sim.populate_connections()
        sim.add_vulnerable_group(.1, .2)
        for i in {0, 2}:
            sim.node_dict[i].active = False
            sim.node_dict[i].output = 0
            sim.node_dict[i].old_active = False
            sim.node_dict[i].old_output = 0
            sim.node_dict[i].undertreated = True
        # No contagion or despair or layoff
        sim.initial_total_output = sim.get_total_output()
        for t in range(time_steps):
            sim.simulation_step(0, 0, num_viral_tests=1, contact_tracing_efficacy=1,
                                hospital_capacity=0.01, undertreatment_effect=100,
                                p_despair_factor=0, subsidy=0, p_inactive_factor=0, record_stats=False)
            assert sim.time == t + 1
            for i in range(sim.num_nodes):
                node = sim.node_dict[i]
                assert not node.infected
                assert not node.deceased
        # Everyone connected to infected gets infected
        print('\nPlease check the details printed')
        sim.time = 0
        initial_infections = np.random.choice(sim.num_nodes, 3, replace=False)
        sim.seed_simulation(initial_infections, record_stats=False)
        sim.stay_at_home_by_occupation_policy(['retiree'])
        for t in range(time_steps):
            print(f'\n====================Day {t}====================')
            print(f'Vulnerability indices: {sim.vulnerability_indices}')
            print(f'Hospitalized nodes: {sim.hospitalized_nodes}')
            print(f'Test backlog: {sim.test_backlog}')
            for i in range(sim.num_nodes):
                node = sim.node_dict[i]
                print(f'\n------------Node {i}------------')
                print(f'Age: {node.age}')
                print(f'Active: {node.active}')
                print(f'Output: {node.output}')
                print(f'Previously active: {node.old_active}')
                print(f'Previous output: {node.old_output}')
                print(f'Household: {node.household}')
                print(f'Household type: {sim.household_dict[node.household]["type"]}')
                print(f'Household members: {sim.household_dict[node.household]["members"]}')
                print(f'Connections: {node.connections}')
                print(f'Vulnerable: {node.vulnerable}')
                print(f'Disease length: {node.disease_length}')
                print(f'Days infected: {node.days_infected}')
                print(f'Symptom date: {node.symptom_date}')
                print(f'Hospital date: {node.hospital_date}')
                print(f'ICU date: {node.ICU_date}')
                print(f'Death date: {node.death_date}')
                print(f'Stay at home: {node.stay_at_home}')
                print(f'Isolated: {node.isolated}')
                print(f'Infected: {node.infected}')
                print(f'Recovered: {node.recovered}')
                print(f'Hospitalized: {node.hospitalized}')
                print(f'Undertreated: {node.undertreated}')
                print(f'ICU: {node.ICU}')
                print(f'Deceased: {node.deceased}')
            sim.simulation_step(1, 1, num_viral_tests=1, contact_tracing_efficacy=1,
                                hospital_capacity=0.01, undertreatment_effect=100,
                                p_despair_factor=.2, subsidy=.2, p_inactive_factor=1, record_stats=False)
            assert sim.time == t + 1

    def test_simulation_step_household_grouping(self):
        time_steps = 10
        average_worker_degree = 2
        sim = Simulation()
        sim.calc_output_measures(average_worker_degree)
        sim.household_network(3, 3, 'poor', 1.0)
        sim.household_network(2, 2, 'rich', 1.0)
        sim.calc_rich_to_poor_output_ratio()
        sim.econ_network(watts_strogatz_graph, [None, average_worker_degree, 0.5])
        sim.populate_connections()
        sim.add_vulnerable_group(.1, .2)
        for i in {0, 2}:
            sim.node_dict[i].active = False
            sim.node_dict[i].output = 0
            sim.node_dict[i].old_active = False
            sim.node_dict[i].old_output = 0
            sim.node_dict[i].undertreated = True
        # No contagion or despair or layoff
        sim.initial_total_output = sim.get_total_output()
        for t in range(time_steps):
            sim.simulation_step(0, 0, num_viral_tests=1, contact_tracing_efficacy=1,
                                hospital_capacity=0.01, undertreatment_effect=100,
                                p_despair_factor=0, subsidy=0, p_inactive_factor=0, record_stats=False)
            assert sim.time == t + 1
            for i in range(sim.num_nodes):
                node = sim.node_dict[i]
                assert not node.infected
                assert not node.deceased
        # Everyone connected to infected gets infected
        print('\nPlease check the details printed')
        sim.time = 0
        initial_infections = np.random.choice(sim.num_nodes, 3, replace=False)
        sim.seed_simulation(initial_infections, record_stats=False)
        sim.stay_at_home_by_occupation_policy(['retiree'])
        for t in range(time_steps):
            print(f'\n====================Day {t}====================')
            print(f'Vulnerability indices: {sim.vulnerability_indices}')
            print(f'Hospitalized nodes: {sim.hospitalized_nodes}')
            print(f'Test backlog: {sim.test_backlog}')
            for i in range(sim.num_nodes):
                node = sim.node_dict[i]
                print(f'\n------------Node {i}------------')
                print(f'Age: {node.age}')
                print(f'Active: {node.active}')
                print(f'Output: {node.output}')
                print(f'Previously active: {node.old_active}')
                print(f'Previous output: {node.old_output}')
                print(f'Household: {node.household}')
                print(f'Household type: {sim.household_dict[node.household]["type"]}')
                print(f'Household members: {sim.household_dict[node.household]["members"]}')
                print(f'Connections: {node.connections}')
                print(f'Vulnerable: {node.vulnerable}')
                print(f'Disease length: {node.disease_length}')
                print(f'Days infected: {node.days_infected}')
                print(f'Symptom date: {node.symptom_date}')
                print(f'Hospital date: {node.hospital_date}')
                print(f'ICU date: {node.ICU_date}')
                print(f'Death date: {node.death_date}')
                print(f'Stay at home: {node.stay_at_home}')
                print(f'Isolated: {node.isolated}')
                print(f'Infected: {node.infected}')
                print(f'Recovered: {node.recovered}')
                print(f'Hospitalized: {node.hospitalized}')
                print(f'Undertreated: {node.undertreated}')
                print(f'ICU: {node.ICU}')
                print(f'Deceased: {node.deceased}')
            sim.simulation_step(1, 1, num_viral_tests={'poor': 1, 'rich': 1}, contact_tracing_efficacy=1,
                                hospital_capacity=0.01, undertreatment_effect=100,
                                p_despair_factor=.2, subsidy=.2, p_inactive_factor=1, record_stats=False)
            assert sim.time == t + 1

    def test_get_total_output_no_household_grouping(self):
        sim = Simulation()
        sim.num_nodes = 5
        sim.household_grouping = False
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        average_worker_degree = 2
        sim.calc_output_measures(average_worker_degree)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            node.active = True
            node.stay_at_home = False
            node.isolated = False
            node.output = Node.personal_output
        sim.node_dict[0].colleagues = [1, 2]
        sim.node_dict[1].colleagues = [0]
        sim.node_dict[2].colleagues = [0, 3]
        sim.node_dict[3].colleagues = [2]
        assert sim.get_total_output() == sim.num_nodes * Node.personal_output + 6 * sim.colleague_linkage_output
        sim.node_dict[0].stay_at_home = True
        assert sim.get_total_output() == sim.num_nodes * Node.personal_output + 2 * sim.colleague_linkage_output
        sim.node_dict[0].stay_at_home = False
        sim.node_dict[0].isolated = True
        assert sim.get_total_output() == sim.num_nodes * Node.personal_output + 2 * sim.colleague_linkage_output
        sim.node_dict[2].active = False
        assert sim.get_total_output() == (sim.num_nodes - 1) * Node.personal_output
        sim.node_dict[0].isolated = False
        assert sim.get_total_output() == ((sim.num_nodes - 1) * Node.personal_output
                                          + 2 * sim.colleague_linkage_output)

    def test_get_total_output_household_grouping(self):
        sim = Simulation()
        sim.num_nodes_poor = 7
        sim.num_nodes_rich = 3
        sim.household_grouping = True
        sim.num_nodes = sim.num_nodes_poor + sim.num_nodes_rich
        sim.node_dict = {i: Node(sample_age()) for i in range(sim.num_nodes)}
        sim.household_dict = {0: {'type': 'rich', 'members': {0, 1, 2}},
                              1: {'type': 'poor', 'members': {3}},
                              2: {'type': 'poor', 'members': {4}},
                              3: {'type': 'poor', 'members': {5, 6, 7, 8, 9}}}
        for i in {0, 1, 2}:
            sim.node_dict[i].household = 0
        sim.node_dict[3].household = 1
        sim.node_dict[4].household = 2
        for i in {5, 6, 7, 8, 9}:
            sim.node_dict[i].household = 3
        average_worker_degree = 2
        sim.calc_output_measures(average_worker_degree)
        for i in range(sim.num_nodes):
            node = sim.node_dict[i]
            node.active = True
            node.stay_at_home = False
            node.isolated = False
            node.output = Node.personal_output
        sim.calc_rich_to_poor_output_ratio()
        sim.node_dict[0].colleagues = [1, 2]
        sim.node_dict[1].colleagues = [0]
        sim.node_dict[2].colleagues = [0, 3, 9]
        sim.node_dict[3].colleagues = [2]
        sim.node_dict[4].colleagues = [5, 7, 8, 9]
        sim.node_dict[5].colleagues = [4]
        sim.node_dict[7].colleagues = [4, 9]
        sim.node_dict[8].colleagues = [4, 9]
        sim.node_dict[9].colleagues = [2, 4, 7, 8]
        total_output_poor = sim.num_nodes_poor * Node.personal_output + 14 * sim.colleague_linkage_output
        total_output_rich = sim.rich_to_poor_output_ratio * (sim.num_nodes_rich * Node.personal_output
                                                             + 6 * sim.colleague_linkage_output)
        assert abs(sim.get_total_output()['poor'] - total_output_poor) < 1e-6
        assert abs(sim.get_total_output()['rich'] - total_output_rich) < 1e-6
        sim.node_dict[0].stay_at_home = True
        total_output_rich = sim.rich_to_poor_output_ratio * (sim.num_nodes_rich * Node.personal_output
                                                             + 2 * sim.colleague_linkage_output)
        assert abs(sim.get_total_output()['poor'] - total_output_poor) < 1e-6
        assert abs(sim.get_total_output()['rich'] - total_output_rich) < 1e-6
        sim.node_dict[0].stay_at_home = False
        sim.node_dict[0].isolated = True
        assert abs(sim.get_total_output()['poor'] - total_output_poor) < 1e-6
        assert abs(sim.get_total_output()['rich'] - total_output_rich) < 1e-6
        sim.node_dict[3].active = False
        total_output_poor = (sim.num_nodes_poor - 1) * Node.personal_output + 13 * sim.colleague_linkage_output
        total_output_rich = sim.rich_to_poor_output_ratio * (sim.num_nodes_rich * Node.personal_output
                                                             + sim.colleague_linkage_output)
        assert abs(sim.get_total_output()['poor'] - total_output_poor) < 1e-6
        assert abs(sim.get_total_output()['rich'] - total_output_rich) < 1e-6
        sim.node_dict[9].stay_at_home = True
        total_output_poor = (sim.num_nodes_poor - 1) * Node.personal_output + 6 * sim.colleague_linkage_output
        total_output_rich = sim.rich_to_poor_output_ratio * sim.num_nodes_rich * Node.personal_output
        assert abs(sim.get_total_output()['poor'] - total_output_poor) < 1e-6
        assert abs(sim.get_total_output()['rich'] - total_output_rich) < 1e-6

    def test_record_aggregate_stats(self):
        assert True

    def test_save_to_csv(self):
        assert True

    def test_get_raw_data(self):
        sim = Simulation()
        infections = [3, 5, 2, 8]
        total_output = {'poor': [0, 1, 3, 9], 'rich': [20, 40, 60, 70]}
        sim.infections = infections
        sim.total_output = total_output
        sim.household_grouping = False
        with pytest.raises(AssertionError, match='Without household grouping, household type must be None'):
            sim.get_raw_data('infections', 'poor')
        assert sim.get_raw_data('infections', None) == infections
        sim.household_grouping = True
        with pytest.raises(AssertionError, match='Household type must be None, "poor", or "rich"'):
            sim.get_raw_data('infections', 'unknown_household_type')
        assert sim.get_raw_data('total_output', 'poor') == total_output['poor']
        assert sim.get_raw_data('total_output', 'rich') == total_output['rich']
        assert sim.get_raw_data('total_output', None) == [total_output['poor'][i] + total_output['rich'][i]
                                                          for i in range(len(total_output['poor']))]

    def test_normalize_no_household_grouping(self):
        time_steps = 10
        sim = Simulation()
        sim.household_network(10, 5, None, 1)
        sim.household_grouping = False
        average_worker_degree = 2
        sim.econ_network(watts_strogatz_graph, [None, average_worker_degree, 0.5])
        sim.populate_connections()
        sim.calc_output_measures(average_worker_degree)
        initial_infections = np.random.choice(sim.num_nodes, 3, replace=False)
        sim.seed_simulation(initial_infections, record_stats=True)
        for _ in range(time_steps):
            sim.simulation_step(0.8, 0.1, 1, record_stats=True)
        stay_at_home_normalized = sim.normalize('stay_at_home_count')
        isolation_normalized = sim.normalize('isolation_count')
        infections_normalized = sim.normalize('infections')
        hospitalizations_cumsum_normalized = sim.normalize('hospitalizations_cumsum')
        active_normalized = sim.normalize('active_count')
        viral_deaths_normalized = sim.normalize('viral_deaths')
        deaths_of_despair_normalized = sim.normalize('deaths_of_despair')
        total_output_normalized = sim.normalize('total_output')
        total_subsidy_normalized = sim.normalize('total_subsidy')
        for t in range(time_steps + 1):
            assert stay_at_home_normalized[t] == sim.stay_at_home_count[t] / sim.num_nodes
            assert isolation_normalized[t] == sim.isolation_count[t] / sim.num_nodes
            assert infections_normalized[t] == sim.infections[t] / sim.num_nodes
            assert hospitalizations_cumsum_normalized[t] == sim.hospitalizations_cumsum[t] / sim.num_nodes
            assert viral_deaths_normalized[t] == sim.viral_deaths[t] / sim.num_nodes
            assert deaths_of_despair_normalized[t] == sim.deaths_of_despair[t] / sim.num_nodes
            assert active_normalized[t] == sim.active_count[t] / sim.num_nodes
            assert total_output_normalized[t] == sim.total_output[t] / sim.total_output[0]
            assert total_subsidy_normalized[t] == sim.total_subsidy[t] / sim.total_output[0]

    def test_normalize_household_grouping(self):
        time_steps = 10
        sim = Simulation()
        sim.household_network(6, 3, 'poor', 1.0)
        sim.household_network(4, 2, 'rich', 1.0)
        sim.calc_rich_to_poor_output_ratio()
        average_worker_degree = 2
        sim.econ_network(watts_strogatz_graph, [None, average_worker_degree, 0.5])
        sim.populate_connections()
        sim.calc_output_measures(average_worker_degree)
        initial_infections = np.random.choice(sim.num_nodes, 3, replace=False)
        sim.seed_simulation(initial_infections, record_stats=True)
        for _ in range(time_steps):
            sim.simulation_step(0.8, 0.1, {'poor': 2, 'rich': 1}, record_stats=True)
        stay_at_home_normalized = sim.normalize('stay_at_home_count', None)
        isolation_normalized = sim.normalize('isolation_count', 'poor')
        infections_normalized = sim.normalize('infections', 'rich')
        hospitalizations_cumsum_normalized = sim.normalize('hospitalizations_cumsum', None)
        active_normalized = sim.normalize('active_count', 'poor')
        viral_deaths_normalized = sim.normalize('viral_deaths', 'rich')
        deaths_of_despair_normalized = sim.normalize('deaths_of_despair', None)
        total_output_normalized_none = sim.normalize('total_output', None)
        total_output_normalized_poor = sim.normalize('total_output', 'poor')
        total_subsidy_normalized_none = sim.normalize('total_subsidy', None)
        total_subsidy_normalized_rich = sim.normalize('total_subsidy', 'rich')
        for t in range(time_steps + 1):
            assert stay_at_home_normalized[t] == (sim.stay_at_home_count['poor'][t]
                                                  + sim.stay_at_home_count['rich'][t]) / sim.num_nodes
            assert isolation_normalized[t] == sim.isolation_count['poor'][t] / sim.num_nodes_poor
            assert infections_normalized[t] == sim.infections['rich'][t] / sim.num_nodes_rich
            assert hospitalizations_cumsum_normalized[t] == (sim.hospitalizations_cumsum['poor'][t]
                                                             + sim.hospitalizations_cumsum['rich'][t]) / sim.num_nodes
            assert active_normalized[t] == sim.active_count['poor'][t] / sim.num_nodes_poor
            assert viral_deaths_normalized[t] == sim.viral_deaths['rich'][t] / sim.num_nodes_rich
            assert deaths_of_despair_normalized[t] == (sim.deaths_of_despair['poor'][t]
                                                       + sim.deaths_of_despair['rich'][t]) / sim.num_nodes
            assert total_output_normalized_none[t] == ((sim.total_output['poor'][t] + sim.total_output['rich'][t])
                                                       / (sim.total_output['poor'][0] + sim.total_output['rich'][0]))
            assert total_output_normalized_poor[t] == sim.total_output['poor'][t] / sim.total_output['poor'][0]
            assert total_subsidy_normalized_none[t] == ((sim.total_subsidy['poor'][t] + sim.total_subsidy['rich'][t])
                                                        / (sim.total_output['poor'][0] + sim.total_output['rich'][0]))
            assert total_subsidy_normalized_rich[t] == sim.total_subsidy['rich'][t] / sim.total_output['rich'][0]

    def test_plot_p_despair(self):
        assert True

    def test_plot_time_series(self):
        assert True

    def test_plot_econ_time_series(self):
        assert True

    def test_plot_all_time_series(self):
        assert True


def test_monte_carlo_save_to_csv():
    assert True


def test_monte_carlo():
    assert True


def test_monte_carlo_read_csv():
    assert True


def test_monte_carlo_single_plot():
    assert True


def test_monte_carlo_multi_plots():
    assert True


def test_monte_carlo_plot_cbar():
    assert True


def test_monte_carlo_contourf():
    assert True
