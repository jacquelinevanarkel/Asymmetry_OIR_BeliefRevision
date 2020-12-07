# Generate coherence networks

# Import packages
import random
import numpy as np

class CoherenceNetworks:

    def __init__(self, number_nodes, amount_edges, amount_positive_constraints):
        """
        Initialisation of class.
        :param number_nodes: int; the number of nodes in the network
        :param amount_edges: string; the amount of edges connecting the nodes in the network (low, middle, high)
        :param amount_positive_constraints: string; the amount of positive constraints connecting the nodes in the
        network (low, middle, high)
        """

        self.n_nodes = number_nodes
        self.a_edges = amount_edges
        self.a_constraint_p = amount_positive_constraints

        # Transform the amount of edges into a randomly chosen number of edges of the associated range
        self.n_edges = self.number_edges()

    def number_edges(self):
        """
        Take the amount of edges (low, middle, high) and turn it into a number of edges.
        :return: int; number of edges
        """

        if self.a_edges == "low":
            n_edges = random.randrange(self.n_nodes-1, 1.5*self.n_nodes)
        elif self.a_edges == "middle":
            n_edges = random.randrange(1.5*self.n_nodes, 2.5*self.n_nodes)
        else:
            n_edges = random.randrange(2.5*self.n_nodes, 4*self.n_nodes)

        return n_edges

    def edges(self):
        """
        Create edges between the nodes.
        :return: array; array containing arrays of two nodes that get connected by an edge
        """

        # First, you want to create random edges so that every node is connected to another one

        # Make sure that all nodes are connected as a network

        # Add leftover edges

        edges = np.array([[0, 1],[]])

        return edges

    def constraints(self):
        """
        Divide the edges into sets of positive and negative constraints.
        :return: array; two arrays containing the positive and negative constraints (edges)
        """





