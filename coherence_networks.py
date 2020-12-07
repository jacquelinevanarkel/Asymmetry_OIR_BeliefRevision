# Generate coherence networks

# Import packages
import random
import networkx as nx
import matplotlib.pyplot as plt

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
        self.n_edges = amount_edges
        self.a_constraint_p = amount_positive_constraints

        # Transform the amount of edges into a randomly chosen number of edges of the associated range
        #self.n_edges = self.number_edges()

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

    def create_graph(self):
        """
        Create a graph based on the number of nodes and amount of edges.
        :return: graph; a graph with the specified number of nodes and edges and added constraints.
        """

        # Create a random graph with n_nodes nodes and n_edges edges
        graph = nx.gnm_random_graph(self.n_nodes, self.n_edges)

        # Add positive or negative constraints to edges
        constraints = random.choices(["positive", "negative"], k=graph.number_of_edges())
        nx.set_edge_attributes(graph, constraints, "constraint")
        colours = ["green" if x == "positive" else "red" for x in constraints]

        nx.draw(graph, edge_color=colours, with_labels=True)
        plt.show()

        return graph

if __name__ == '__main__':
    CoherenceNetworks(10, 15, 4).create_graph()




