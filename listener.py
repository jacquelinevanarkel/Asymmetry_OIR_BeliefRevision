# Import packages
import networkx as nx
from coherence_networks import CoherenceNetworks
import random

class ListenerModel:

    def __init__(self, belief_network, node_type, node_truth_value):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        """

        # Initialise the different sets of nodes (own, communicated, inferred)
        self.belief_network = belief_network
        self.node_type = node_type
        self.node_truth_value = node_truth_value

        # Initialise a network history, which stores the network one time step back (so before the belief revision takes
        # place)
        self.network_history = belief_network

        # Add the truth values and the type to the nodes in the belief network
        for i in range(len(node_truth_value)):
            self.belief_network.nodes[i]['truth_value'] = self.node_truth_value[i]
            self.belief_network.nodes[i]['type'] = self.node_type[i]
        # print(self.belief_network.nodes(data=True))

    def belief_revision(self):
        """
        Make inferences that maximise coherence after new node(s) have been communicated.
        :return: graph; the new belief network
        """

        gain_coherence = []
        for node in self.belief_network.nodes(data=True):
            # print(node)
            if node[1]['type'] == None or node[1]['type'] == 'inf':
                if node[1]['truth_value'] == None:
                    node[1]['truth_value'] = random.choice([True, False])
                print(self.belief_network.nodes(data=True))
                # Greedy algorithm?
                coherence = self.coherence()
                node[1]['truth_value'] = not node[1]['truth_value']
                # print(self.belief_network.nodes(data=True))
                coherence_new = self.coherence()
                gain_coherence.append((node[0], coherence_new - coherence))
                node[1]['truth_value'] = not node[1]['truth_value']
                # print("Coherence= ", coherence)
                # print("Coherence new = ", coherence_new)
                # if coherence_new <= coherence:
                #     node[1]['truth_value'] = not node[1]['truth_value']
        print(self.belief_network.nodes(data=True))
        print(gain_coherence)
        node_flipped = max(gain_coherence, key=lambda x: x[1])
        print(node_flipped)
        if node_flipped[1] > 0:
            self.belief_network.nodes[node_flipped[0]]['truth_value'] = not self.belief_network.nodes[node_flipped[0]]['truth_value']
        print(self.belief_network.nodes(data=True))

        return belief_network

    def coherence(self):
        """
        Calculate the coherence of a belief network.
        :return: float; the coherence of a belief network
        """

        # Initialise coherence
        coherence = 0

        for edge in list(self.belief_network.edges(data='constraint')):
            if edge[2] == 'positive':
                if self.belief_network.nodes[edge[0]]['truth_value'] == self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence += 1
                else:
                    coherence -= 1
            elif edge[2] == 'negative':
                if self.belief_network.nodes[edge[0]]['truth_value'] == self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence -= 1
                else:
                    coherence += 1
        # print("Coherence = ", coherence)
        return coherence

    def trouble_identification(self):
        """
        Compares the coherence of the current belief network with the previous one (before last communicated node(s)).
        :return: boolean; whether to initiate repair or not
        """

    def formulate_request(self):
        """
        Decide which node(s) to include into the restricted offer, based on the truth value flip(s) that results in the
        highest coherence.
        :return: list; the node(s) included in the restricted offer
        """

if __name__ == '__main__':
    belief_network = CoherenceNetworks(5, 'middle', 'middle').create_graph()
    # ListenerModel(belief_network, ['own', 'own', 'com', None, None], [False, True, True, False, False]).coherence()
    ListenerModel(belief_network, ['own', 'own', 'com', None, 'inf'], [False, True, True, False, False]).belief_revision()