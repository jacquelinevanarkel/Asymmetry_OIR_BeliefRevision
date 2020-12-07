# Import packages
import numpy as np

class Interpretation:

    def __init__(self, relevant, own, communicated, constraint_pos, constraint_neg):
        """
        Initialisation of class.
        :param relevant: array; the set of relevant nodes
        :param own: array; the set of own beliefs in the network of relevant nodes
        :param communicated: array; the set of communicated beliefs in the network of relevant nodes
        :param constraint_pos: array; the set of edges with a positive constraint
        :param constraint_neg: array; the set of edges with a negative constraint
        """

        # Initialise the different sets of nodes (relevant, own, communicated, inferred)
        nodes_rel = np.array(relevant)
        nodes_own = np.array(own)
        nodes_comm = np.array(communicated)
        nodes_inf = np.array([])