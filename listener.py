# Import packages
import networkx as nx
from coherence_networks import CoherenceNetworks
import random
import itertools


class ListenerModel:

    def __init__(self, belief_network, node_type, node_truth_value, communicated_nodes=None, init=True):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network (None if not specified)
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        :param communicated_nodes: list; a list containing the truth values of the node(s) communicated by the speaker
        """

        if init:
            # Initialise the different sets of nodes (own, communicated, inferred)
            self.belief_network = belief_network
            self.node_type = node_type
            self.node_truth_value = node_truth_value
            self.communicated_nodes = communicated_nodes

            # Initialise a coherence history, which stores the coherence of the network one time step back (so before
            # the belief revision takes place)
            self.coherence_history = None

            # If a type value is none, set it to inferred so they can be inferred in belief_revision()
            self.node_type = ['inf' if type is None else type for type in self.node_type]

            # Add the truth values and the type to the nodes in the belief network
            for i in range(len(node_truth_value)):
                self.belief_network.nodes[i]['truth_value'] = self.node_truth_value[i]
                self.belief_network.nodes[i]['type'] = self.node_type[i]
            nx.set_node_attributes(self.belief_network, False, "repair")

        print("Network at start \n", self.belief_network.nodes(data=True))

    def belief_revision(self):
        """
        Make inferences that maximise coherence after new node(s) have been communicated.
        """

        # Store the coherence of the network before the belief revision has taken place
        self.coherence_history = self.coherence(self.belief_network)
        network_history = self.belief_network.copy()

        # Add the newly communicated nodes to the network
        if self.communicated_nodes is not None:
            for node in self.communicated_nodes:
                self.belief_network.nodes[node[0]]['truth_value'] = node[1]
                self.belief_network.nodes[node[0]]['type'] = 'com'
                print("Coherence after com: ", self.coherence(self.belief_network))

        print("Network before belief revision \n", self.belief_network.nodes(data=True))

        # Get the inferred nodes and its combinations of truth values in order to explore different coherence values
        inferred_nodes = [x for x, y in self.belief_network.nodes(data=True) if y['type'] == 'inf']
        print(inferred_nodes)
        combinations = list(itertools.product([True, False], repeat=len(inferred_nodes)))
        print(combinations)

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

        print("coherence_values: ", coherence_values)

        # Store all the indices of the maximum coherence values in a list and pick one randomly
        max_coherence = max(coherence_values)
        max_indices = [i for i in range(len(coherence_values)) if coherence_values[i] == max_coherence]
        nodes_truth_values_index = random.choice(max_indices)

        print("max indices: ", max_indices)
        print("nodes_truth_values_index: ", nodes_truth_values_index)

        # Set the truth values of the inferred nodes to (one of) the maximum coherence option(s)
        i = 0
        for inferred_node in inferred_nodes:
            self.belief_network.nodes[inferred_node]['truth_value'] = combinations[nodes_truth_values_index][i]
            i += 1
        print(self.belief_network.nodes(data=True))

        # If at least one node is flipped, belief revision has taken place and the coherence should be compared
        # with the previous network before belief revision (trouble_identification)
        if not nx.is_isomorphic(self.belief_network, network_history, node_match=lambda x, y: x['truth_value'] ==
                                                                                                   y['truth_value']):
            print("Network after belief revision: \n", self.belief_network.nodes(data=True))
            print("Trouble identification")
            self.trouble_identification()

        return self.belief_network

    def coherence(self, network):
        """
        Calculate the coherence of a belief network.
        :return: float; the coherence of a belief network
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
        # print("Coherence = ", coherence)
        return coherence

    def trouble_identification(self):
        """
        Compares the coherence of the current belief network with the previous one (before last communicated node(s)).
        """

        print("coherence history:", self.coherence_history)
        print("coherence:", self.coherence(self.belief_network))
        # Initiate repair if the coherence is smaller than one time step back
        if self.coherence(self.belief_network) < self.coherence_history:
            self.formulate_request()
            print("REPAIR!")
        else:
            return None

    def formulate_request(self):
        """
        Decide which node(s) to include into the restricted offer, based on the truth value flip(s) that results in the
        highest coherence.
        :return: list; the node(s) included in the restricted offer
        """

        # Initialise a list to store the indices of the nodes to include in the repair initiation
        repair_initiation = []

        # Make a copy of the existing belief network in order to explore the coherence of different truth value
        # assignment combinations
        network_copy = self.belief_network.copy()

        print("Formulate repair, network state:\n", self.belief_network.nodes(data=True))

        # Get the not (yet) communicated nodes and its combinations of truth values in order to explore
        # different coherence values
        not_comm_nodes = [x for x, y in network_copy.nodes(data=True) if y['type'] == 'inf' or
                          y['type'] == 'own' and y['repair'] is False]
        print(not_comm_nodes)
        combinations = list(itertools.product([True, False], repeat=len(not_comm_nodes)))
        print(combinations)

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

        print("coherence_values: ", coherence_values)

        # Store all the indices of the maximum coherence values in a list and pick one randomly
        max_coherence = max(coherence_values)
        max_indices = [i for i in range(len(coherence_values)) if coherence_values[i] == max_coherence]
        nodes_truth_values_index = random.choice(max_indices)

        print("max indices: ", max_indices)
        print("nodes_truth_values_index: ", nodes_truth_values_index)

        # The node(s) to be asked repair over are stored in a list containing tuples (a tuple per node to ask repair
        # over) consisting of the node index and its truth value
        for not_comm_node in not_comm_nodes:
            if network_copy.nodes[not_comm_node]['truth_value'] != self.belief_network.nodes[not_comm_node]['truth_value']:
                repair_initiation.append((not_comm_node, self.belief_network.nodes[not_comm_node]['truth_value']))
                self.belief_network.nodes[not_comm_node]['repair'] = True

        print("repair:", repair_initiation)
        return repair_initiation


if __name__ == '__main__':
    belief_network = CoherenceNetworks(10, 'high', 'middle').create_graph()
    # ListenerModel(belief_network, ['own', 'own', 'com', None, None], [False, True, True, False, False]).coherence()
    # ListenerModel(belief_network, ['own', 'own', 'com', None, 'inf'], [False, True, True, True, False]).belief_revision()
    # ListenerModel(belief_network, ['own', 'own', 'com', None, 'inf', 'own', 'own', 'com', None, 'inf'],
    #               [False, True, True, True, False, False, True, True, True, False],
    #               communicated_nodes=[None, False, None, True, None, None, None, None, None, None]).belief_revision()
    ListenerModel(belief_network, ['own', 'own', 'com', None, 'inf', 'own', 'own', 'com', None, 'inf'],
                  [False, True, True, True, False, False, True, True, True, False]).belief_revision()
