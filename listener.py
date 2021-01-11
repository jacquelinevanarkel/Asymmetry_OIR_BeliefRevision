# Import packages
import networkx as nx
from coherence_networks import CoherenceNetworks
import random
import itertools


class ListenerModel:

    def __init__(self, belief_network, communicated_nodes=None):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param communicated_nodes: list; a list containing the truth values of the node(s) communicated by the speaker
        """

        # Initialise the belief belief_network and the communicated nodes
        self.belief_network = belief_network
        self.communicated_nodes = communicated_nodes

        # Initialise a coherence history, which stores the coherence of the belief_network one time step back (so before
        # the belief revision takes place)
        self.coherence_history = None

    def belief_revision(self):
        """
        Make inferences that maximise coherence after new node(s) have been communicated.
        """

        # Store the coherence of the belief_network before the belief revision has taken place
        network_history = self.belief_network.copy()
        self.coherence_history = self.coherence(network_history)

        # Add the newly communicated nodes to the belief_network
        if self.communicated_nodes is not None:
            for node in self.communicated_nodes:
                self.belief_network.nodes[node[0]]['truth_value'] = node[1]
                self.belief_network.nodes[node[0]]['type'] = 'com'

        # Get the inferred nodes and its combinations of truth values in order to explore different coherence values
        inferred_nodes = [x for x, y in self.belief_network.nodes(data=True) if y['type'] == 'inf']
        combinations = list(itertools.product([True, False], repeat=len(inferred_nodes)))

        # Calculate the coherence for all possible combinations

        # Initialise a list to store the different coherence values in
        coherence_values = []

        for n in range(len(combinations)):
            # Initialise a count for the number of inferred nodes
            i = 0
            for inferred_node in inferred_nodes:
                self.belief_network.nodes[inferred_node]['truth_value'] = combinations[n][i]
                i += 1
            coherence_values.append(self.coherence(self.belief_network))

        # Store all the indices of the maximum coherence values in a list and pick one randomly
        max_coherence = max(coherence_values)
        max_indices = [i for i in range(len(coherence_values)) if coherence_values[i] == max_coherence]
        nodes_truth_values_index = random.choice(max_indices)

        # Set the truth values of the inferred nodes to (one of) the maximum coherence option(s)
        i = 0
        for inferred_node in inferred_nodes:
            self.belief_network.nodes[inferred_node]['truth_value'] = combinations[nodes_truth_values_index][i]
            i += 1

        # If at least one node is flipped, belief revision has taken place and the coherence should be compared
        # with the previous belief_network before belief revision (trouble_identification)
        print("Network after belief revision:\n", self.belief_network.nodes(data=True))
        print("Network before belief revision:\n", network_history.nodes(data=True))
        if not nx.is_isomorphic(self.belief_network, network_history, node_match=lambda x, y: x['truth_value'] ==
                                                                                                   y['truth_value']):
            print("Trouble identification")
            repair_initiation = self.trouble_identification()
        else:
            print("No trouble identification")
            repair_initiation = False

        return repair_initiation, self.belief_network

    def coherence(self, network):
        """
        Calculate the coherence of a belief belief_network.
        :return: float; the coherence of a belief belief_network
        """

        # Initialise coherence
        coherence = 0

        for edge in list(network.edges(data='constraint')):
            if edge[2] == 'positive':
                if network.nodes[edge[0]]['truth_value'] == \
                        network.nodes[edge[1]]['truth_value']:
                    coherence += 1
                else:
                    coherence -= 1
            elif edge[2] == 'negative':
                if network.nodes[edge[0]]['truth_value'] == \
                        network.nodes[edge[1]]['truth_value']:
                    coherence -= 1
                else:
                    coherence += 1

        return coherence

    def trouble_identification(self):
        """
        Compares the coherence of the current belief belief_network with the previous one (before last communicated node(s)).
        """

        # Initiate repair if the coherence is smaller than one time step back
        if self.coherence(self.belief_network) < self.coherence_history:
            repair_initiation = self.formulate_request()
        else:
            print("Not formulating a request")
            repair_initiation = False

        return repair_initiation

    def formulate_request(self):
        """
        Decide which node(s) to include into the restricted offer, based on the truth value flip(s) that results in the
        highest coherence.
        :return: list; the node(s) included in the restricted offer
        """

        # Initialise a list to store the indices of the nodes to include in the repair initiation
        repair_initiation = []

        # Make a copy of the existing belief belief_network in order to explore the coherence of different truth value
        # assignment combinations
        network_copy = self.belief_network.copy()

        # Get the not (yet) communicated nodes and its combinations of truth values in order to explore
        # different coherence values
        not_comm_nodes = [x for x, y in network_copy.nodes(data=True) if y['type'] == 'inf' or
                          y['type'] == 'own' and y['repair'] is False]

        # If there are no nodes that have not been communicated yet, break and return 'False' as a repair initiation
        if not_comm_nodes is None:
            return False

        combinations = list(itertools.product([True, False], repeat=len(not_comm_nodes)))

        # Calculate the coherence for all possible combinations

        # Initialise a list to store the different coherence values in
        coherence_values = []

        for n in range(len(combinations)):
            # Initialise a count for the number of inferred nodes
            i = 0
            for not_comm_node in not_comm_nodes:
                network_copy.nodes[not_comm_node]['truth_value'] = combinations[n][i]
                i += 1
            coherence_values.append(self.coherence(network_copy))

        # Store all the indices of the maximum coherence values in a list and pick one randomly
        max_coherence = max(coherence_values)
        max_indices = [i for i in range(len(coherence_values)) if coherence_values[i] == max_coherence]
        nodes_truth_values_index = random.choice(max_indices)

        # Change the network copy to the truth value combination with the highest coherence
        i = 0
        for not_comm_node in not_comm_nodes:
            network_copy.nodes[not_comm_node]['truth_value'] = combinations[nodes_truth_values_index][i]
            i += 1

        # The node(s) to be asked repair over are stored in a list containing tuples (a tuple per node to ask repair
        # over) consisting of the node index and its truth value
        for not_comm_node in not_comm_nodes:
            if network_copy.nodes[not_comm_node]['truth_value'] != self.belief_network.nodes[not_comm_node]['truth_value']:
                repair_initiation.append((not_comm_node, self.belief_network.nodes[not_comm_node]['truth_value']))
                self.belief_network.nodes[not_comm_node]['repair'] = True

        return repair_initiation
