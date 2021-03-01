# Import packages
import random
import networkx as nx
import matplotlib.pyplot as plt


class CoherenceNetworks:

    def __init__(self, number_nodes, amount_edges, amount_positive_constraints):
        """
        Initialisation of class.
        :param number_nodes: int; the number of nodes in the belief_network
        :param amount_edges: string; the amount of edges connecting the nodes in the belief_network (low, middle, high)
        :param amount_positive_constraints: string; the amount of positive constraints connecting the nodes in the
        belief_network (low, middle, high)
        """

        self.n_nodes = number_nodes
        self.a_edges = amount_edges
        self.a_constraint_pos = amount_positive_constraints

        # Transform the amount of edges into a randomly chosen number of edges of the associated range
        self.n_edges = self.number_edges()

        # Transform the amount of positive constraints into a randomly chosen probability of the associated range
        self.prob_pos_edges = self.probability_constraint_p()

    def number_edges(self):
        """
        Take the amount of edges (low, middle, high) and turn it into a number of edges by using fractions of the upper
        bound.
        :return: int; number of edges
        """

        # The maximum number of edges is (the number of nodes*(number of nodes -1))/2, forming an upper bound.
        if self.a_edges == "low":
            n_edges = round(random.uniform(((self.n_nodes - 1)/(self.n_nodes**2)), 0.333) * ((self.n_nodes*(self.n_nodes - 1))/2))
        elif self.a_edges == "middle":
            n_edges = round(random.uniform(0.34, 0.666) * (self.n_nodes*(self.n_nodes - 1))/2)
        elif self.a_edges == "high":
            n_edges = round(random.uniform(0.67, 1.0) * (self.n_nodes*(self.n_nodes - 1))/2)
        else:
            raise ValueError("Amount of edges must be either 'low', 'middle' or 'high'")

        return n_edges

    def probability_constraint_p(self):
        """
        Transforms input of 'low', 'middle', 'high' in a random chosen probability within the specified range to choose
        a positive constraint over a negative one.
        :return: int; probability to choose a positive constraint
        """

        if self.a_constraint_pos == "low":
            prob_pos_edges = random.uniform(0, 0.333)
        elif self.a_constraint_pos == "middle":
            prob_pos_edges = random.uniform(0.34, 0.666)
        elif self.a_constraint_pos == "high":
            prob_pos_edges = random.uniform(0.67, 1)
        else:
            raise ValueError("Amount of positive constraints must be either 'low', 'middle' or 'high'")

        return prob_pos_edges

    def create_graph(self):
        """
        Create a graph based on the number of nodes, amount of edges and amount of positive constraints.
        :return: graph; a graph with the specified number of nodes and edges and added constraints.
        """

        # Create a random graph with n_nodes nodes and n_edges edges
        graph = nx.gnm_random_graph(self.n_nodes, self.n_edges)

        # Make sure that the graph is connected (meaning no separate graphs)
        while nx.is_connected(graph) == False:
            graph = nx.gnm_random_graph(self.n_nodes, self.n_edges)

        # Add positive or negative constraints to edges
        constraints = random.choices(["positive", "negative"], weights=[self.prob_pos_edges, 1 - self.prob_pos_edges],
                                     k=graph.number_of_edges())
        i = 0
        for edge in list(graph.edges()):
            graph[edge[0]][edge[1]]["constraint"] = constraints[i]
            i += 1
        # print(constraints)
        # print(graph.edges(data=True))
        # nx.set_edge_attributes(graph, constraints, "constraint")
        # print("Prob pos edges: ", self.prob_pos_edges)
        # print("Number of edges: ", self.n_edges)

        # Draw the graph
        # colours = ["green" if x == "positive" else "red" for x in constraints]
        # nx.draw(graph, edge_color=colours, with_labels=True)
        # plt.show()

        return graph


if __name__ == '__main__':
    CoherenceNetworks(8, 'middle', 'high').create_graph()
