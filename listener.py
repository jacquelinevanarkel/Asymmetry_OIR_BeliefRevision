# Import packages
import networkx as nx
from coherence_networks import CoherenceNetworks

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

        # Add the truth values to the nodes in the belief network
        for i in range(len(node_truth_value)):
            self.belief_network.nodes[i]['truth_value'] = node_truth_value[i]
        # print(self.belief_network.nodes(data=True))

    def belief_revision(self):
        """
        Make inferences that maximise coherence after new node(s) have been communicated.
        :return: graph; the new belief network
        """

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
    ListenerModel(belief_network, [None, None, None, None, None], [False, True, True, False, False]).coherence()