# Import packages
import numpy as np
import networkx as nx

class BeliefRevision:

    def __init__(self, belief_network, node_type, node_truth_value):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        """

        # Initialise the different sets of nodes (own, communicated, inferred)
        self.nodes_own = np.array(own)
        self.nodes_comm = np.array(communicated)
        self.nodes_inf = np.array(inferred)

        # Initialise a network history, which stores the network one time step back (so before the belief revision takes
        # place)
        self.network_history = belief_network

    def inferring(self):

        return belief_network

    def coherence(self):

        return coherence

class TroubleIdentification:

    def __init__(self, belief_network, node_type, node_truth_value, coherence_previous):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        :param coherence_previous: float; the coherence of the previous belief network (before the last nodes that were
        communicated)
        """

class RequestFormulation:

    def __init__(self):