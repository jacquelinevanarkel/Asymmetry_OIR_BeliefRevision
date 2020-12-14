# Import packages
import networkx as nx

class SpeakerModel:

    def __init__(self, belief_network, node_truth_value):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints
        :param node_truth_value: list; a list containing the truth values of all the nodes in the network
        """

        self.belief_network = belief_network
        self.node_truth_value = node_truth_value

    def communicate_belief(self):
        """
        Communicate a (set of) node(s), which will be chosen randomly. Already communicated nodes can't be communicated
        again.
        :return: list; the truth values of a (set of) node(s) to be communicated
        """

        # https://docs.python.org/3/library/itertools.html --> look at subsets enumereren

    def belief_revision(self):
        """
        Make inferences based on own and communicated beliefs (same function as for listener).
        :return: list; the truth values of a set of inferred nodes
        """

    def repair_solution(self):
        """
        Confirm/disconfirm the restricted offer and add clarification if needed.
        :return: list; the truth values of a (set of) node(s), at least including the truth value of the restricted
        offer.
        """

    def coherence(self):
        """
        Calculate coherence of a belief network.
        :return: float; coherence
        """