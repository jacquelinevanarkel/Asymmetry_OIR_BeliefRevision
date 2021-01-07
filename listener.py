# Import packages
import networkx as nx
from coherence_networks import CoherenceNetworks
import random


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
            if communicated_nodes is None:
                communicated_nodes = [None]
            self.communicated_nodes = communicated_nodes

            # Initialise a network history, which stores the coherence of the network one time step back (so before the
            # belief revision takes place)
            self.network_history = None

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
        self.network_history = self.coherence()

        # Initialise belief_revision: whether belief revision has taken place
        belief_revision = False

        # Add the newly communicated nodes to the network
        for node in self.communicated_nodes:
            self.belief_network.nodes[node[0]]['truth_value'] = node[1]
            self.belief_network.nodes[node[0]]['type'] = 'com'

        print("Network before belief revision \n", self.belief_network.nodes(data=True))

        # TODO: change to exhaustive search (use itertools to see which combination of all combinations has highest
        #  coherence)
        # Greedy algorithm: flip the node (inferred or None) that has the highest gain in coherence
        # Repeat for 2 times the number of nodes in the network
        for _ in range(2 * self.belief_network.number_of_nodes()):
            # Initialise a list that keeps track of the gain of coherence when the node would be flipped for all the
            # nodes (inferred or None) in the network
            gain_coherence = []

            # Check which node flip increases coherence most of the possible nodes in the network (inferred or None)
            for node in self.belief_network.nodes(data=True):

                # When the truth value is set to None, initialise it with a random truth value and set the type to
                # inferred
                if node[1]['type'] is None:
                    node[1]['truth_value'] = random.choice([True, False])
                    node[1]['type'] = 'inf'

                # Only flip inferred nodes
                if node[1]['type'] == 'inf':
                    # Calculate the coherence before the node is flipped
                    coherence = self.coherence()

                    # Calculate the coherence when the node is flipped and store in the list of coherence gain (together
                    # with the node index)
                    node[1]['truth_value'] = not node[1]['truth_value']
                    coherence_new = self.coherence()
                    gain_coherence.append((node[0], coherence_new - coherence))

                    # Flip the truth value of the node back to its original truth value
                    node[1]['truth_value'] = not node[1]['truth_value']

            # Check which flip of the truth value of a node has the highest gain in coherence and flip its truth value
            # when the gain in coherence is higher than 0

            # First the highest gain in coherence is found in the array containing tuples of all the gains of coherence
            # and their corresponding node indices
            node_flipped = max(gain_coherence, key=lambda x: x[1])

            # If the highest gain in coherence is bigger than 0, the corresponding node is flipped
            if node_flipped[1] > 0:
                print("Node belief revision: ", node_flipped[0], "gain: ", node_flipped[1])
                self.belief_network.nodes[node_flipped[0]]['truth_value'] = \
                    not self.belief_network.nodes[node_flipped[0]]['truth_value']
                # If at least one node is flipped, belief revision has taken place and the coherence should be compared
                # with the previous network before belief revision.
                belief_revision = True

        if belief_revision:
            print("Network after belief revision: \n", self.belief_network.nodes(data=True))
            print("Trouble identification")
            self.trouble_identification()

        return self.belief_network

    def coherence(self):
        """
        Calculate the coherence of a belief network.
        :return: float; the coherence of a belief network
        """

        # Initialise coherence
        coherence = 0

        for edge in list(self.belief_network.edges(data='constraint')):
            if edge[2] == 'positive':
                if self.belief_network.nodes[edge[0]]['truth_value'] == \
                        self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence += 1
                else:
                    coherence -= 1
            elif edge[2] == 'negative':
                if self.belief_network.nodes[edge[0]]['truth_value'] == \
                        self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence -= 1
                else:
                    coherence += 1
        # print("Coherence = ", coherence)
        return coherence

    def trouble_identification(self):
        """
        Compares the coherence of the current belief network with the previous one (before last communicated node(s)).
        """

        # Initiate repair if the coherence is smaller than one time step back
        if self.coherence() < self.network_history:
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

        # Initialise a list to store the indeces of the nodes to include in the repair initiation
        repair_initiation = []

        print("Formulate repair, network state:\n", self.belief_network.nodes(data=True))

        # TODO: change to exhaustive search (use itertools to see which combination of all combinations has highest
        #  coherence)
        # Greedy algorithm: flip the node (own or inferred) that has the highest gain in coherence
        # Repeat for 2 times the number of nodes in the network
        for _ in range(2 * self.belief_network.number_of_nodes()):
            # Initialise a list that keeps track of the gain of coherence when the node would be flipped for all the
            # nodes (own or inferred) in the network
            gain_coherence = []

            # Check which node flip increases coherence most of the possible nodes in the network (inferred or None)
            for node in self.belief_network.nodes(data=True):

                # Only flip inferred nodes and nodes without a type yet and make sure not to ask repair about the same
                # node
                if (node[1]['type'] == 'own' or node[1]['type'] == 'inf') and node[1]['repair'] is False:
                    # Calculate the coherence before the node is flipped
                    coherence = self.coherence()

                    # Calculate the coherence when the node is flipped and store in the list of coherence gain (together
                    # with the node index and its original truth value)
                    node[1]['truth_value'] = not node[1]['truth_value']
                    coherence_new = self.coherence()
                    gain_coherence.append((node[0], coherence_new - coherence, not node[1]['truth_value']))

                    # Flip the truth value of the node back to its original truth value
                    node[1]['truth_value'] = not node[1]['truth_value']

            # Add the index and truth value of a node to the list if the flip of the truth value of that node has the
            # highest gain in coherence and the gain in coherence is higher than 0

            # First the highest gain in coherence is found in the array containing tuples of all the gains of coherence
            # and their corresponding node indices and truth values
            node_flipped = max(gain_coherence, key=lambda x: x[1])

            # If the highest gain in coherence is bigger than 0, the corresponding node is saved in a list to ask repair
            # over. This list contains tuples (a tuple per node to ask repair over) consisting of the node index and
            # its truth value
            if node_flipped[1] > 0:
                print("Gain coherence: ", gain_coherence)
                print("Node flipped coherence gain: ", node_flipped[1])
                repair_initiation.append((node_flipped[0], node_flipped[2]))
                self.belief_network.nodes[node_flipped[0]]['repair'] = True

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
                  [False, True, True, True, False, False, True, True, True, False],
                  communicated_nodes=[(0, True), (5, False)]).belief_revision()
