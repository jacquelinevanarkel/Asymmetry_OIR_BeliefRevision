# Import packages
import networkx as nx
import random
from coherence_networks import CoherenceNetworks
import itertools

class SpeakerModel:

    def __init__(self, belief_network, node_type, node_truth_value, intention, repair_request=None, init=True):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network (None if not specified)
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        :param repair_request: list; list of tuples with the nodes (index, truth value) over which repair is asked
        :param intention: list; list with the indices of the nodes that form the speaker's intention
        """

        if init:
            self.belief_network = belief_network
            self.node_type = node_type
            self.node_truth_value = node_truth_value
            self.repair_request = repair_request
            self.intention = intention

            # Initialise a similarity score to keep track of the similarity between the inferred intention and the
            # speaker's intention
            self.similarity = 0

            # If a type value is none, set it to inferred so they can be inferred in belief_revision()
            self.node_type = ['inf' if type is None else type for type in self.node_type]

            # Add the truth values and the type to the nodes in the belief network
            for i in range(len(node_truth_value)):
                self.belief_network.nodes[i]['truth_value'] = self.node_truth_value[i]
                self.belief_network.nodes[i]['type'] = self.node_type[i]

            # Initialise all nodes with communicated set to false, to keep track of which nodes are already communicated
            # by the speaker
            nx.set_node_attributes(self.belief_network, False, "communicated")

    def communicate_beliefs(self):
        """
        Communicate a (set of) node(s), which will be chosen such that when a belief revision is performed
        (with an egocentric listener model) it is most similar to the communicative intentions and the utterance should
        be as short as possible. Already communicated nodes can't be communicated again.
        :return: list; the truth values of a (set of) node(s) to be communicated (the other nodes will be set to 'None')
        """

        # First of all, the speaker makes inferences about the nodes that are not its own beliefs or the communicative
        # intention (T'_inf)
        print("Belief network before belief revision: ", self.belief_network.nodes(data=True))
        self.belief_network = self.belief_revision(self.belief_network)
        print("Belief network after belief revision: ", self.belief_network.nodes(data=True))

        # This part should be done in a loop: for i in range(n_nodes) --> combinations(nodes, i) where nodes are nodes
        # that haven't been communicated yet

        # Get the not (yet) communicated nodes and its combinations of different sizes of (sub)sets
        not_comm_nodes = [x for x, y in self.belief_network.nodes(data=True) if y['type'] == 'inf' or
                          y['type'] == 'own']
        print("Nodes that can be communicated: ", not_comm_nodes)

        # For every combination of (subsets) of nodes, calculate the similarity after belief revision and divide over
        # the number of nodes

        # Initialise a list for combinations of (sub)sets of nodes to be uttered
        combinations = []

        # Add all those (sub)sets to a list of combinations
        for r in range(1, len(not_comm_nodes)+1):
            combinations.extend(list(itertools.combinations(not_comm_nodes, r)))
        print("Combinations: ", combinations)

        # Perform belief revision for every combination and calculate the similarity and divide over the number of nodes
        optimisation = []
        for combination in combinations:
            network = self.belief_network.copy()
            network_listener = self.belief_revision(network, communicated_nodes=combination)

            # Calculate similarity
            similarity = 0
            for node in self.intention:
                if network_listener.nodes[node]['truth_value'] == self.belief_network.nodes[node]['truth_value']:
                    similarity += 1
            optimisation.append(similarity/len(combination))

        print("Optimisation: ", optimisation)
        # The utterance is the combination with the highest optimisation
        max_optimisation = max(optimisation)
        max_indices = [i for i in range(len(optimisation)) if optimisation[i] == max_optimisation]
        optimisation_index = random.choice(max_indices)
        utterance_indices = combinations[optimisation_index]
        print("Optimisation index: ", optimisation_index)
        print("Utterance indices: ", utterance_indices)

        utterance = []
        for index in utterance_indices:
            utterance.append((index,self.belief_network.nodes[index]['truth_value']))

        print("Utterance: ", utterance)

        return utterance, self.belief_network

    def belief_revision(self, network, communicated_nodes=None):
        """
        Make inferences based on own and communicated beliefs (same function as for listener).
        :return: list; the truth values of a set of inferred nodes
        """

        # Add communicated nodes if necessary
        if communicated_nodes is not None:
            for node in communicated_nodes:
                network.nodes[node]['truth_value'] = self.belief_network.nodes[node]['truth_value']
                network.nodes[node]['type'] = 'com'

        # Get the inferred nodes and its combinations of truth values in order to explore different coherence values
        inferred_nodes = [x for x, y in network.nodes(data=True) if y['type'] == 'inf']
        combinations = list(itertools.product([True, False], repeat=len(inferred_nodes)))

        # Calculate the coherence for all possible combinations

        # Initialise a list to store the different coherence values in
        coherence_values = []

        for n in range(len(combinations)):
            # Initialise a count for the number of inferred nodes
            i = 0
            for inferred_node in inferred_nodes:
                network.nodes[inferred_node]['truth_value'] = combinations[n][i]
                i += 1
            coherence_values.append(self.coherence(network))

        # Store all the indices of the maximum coherence values in a list and pick one randomly
        max_coherence = max(coherence_values)
        max_indices = [i for i in range(len(coherence_values)) if coherence_values[i] == max_coherence]
        nodes_truth_values_index = random.choice(max_indices)

        # Set the truth values of the inferred nodes to (one of) the maximum coherence option(s)
        i = 0
        for inferred_node in inferred_nodes:
            network.nodes[inferred_node]['truth_value'] = combinations[nodes_truth_values_index][i]
            i += 1

        return network

    def repair_solution(self):
        """
        Confirm/disconfirm the restricted offer and add clarification if needed.
        :return: list; the indices and speaker's truth values of the repair request and an additional list of the
        indices and truth values of node(s) when no confirmation could be given, else these lists are empty
        """

        # Check whether the truth values of repair request match with the speaker's network, if not, no confirmation can
        # be given
        confirmation = True
        for node in self.repair_request:
            if node[1] != self.belief_network.nodes[node[0]]['truth_value']:
                confirmation = False
                break

        # If no confirmation can be given, the speaker communicates their own truth values of the repair request and
        # gives an additional clarification
        repair = []
        clarification = None
        if not confirmation:
            for node in self.repair_request:
                repair.append((node[0], self.belief_network.nodes[node[0]]['truth_value']))
            #clarification = self.communicate_belief()
            clarification = []

        repair_solution = repair + clarification

        return repair_solution

    # def end_conversation(self):
    #     """
    #     End the conversation if no repair is initiated anymore and the speaker has communicated its intentions.
    #     :return: boolean; true if the conversation should be ended, false otherwise
    #     """
    #
    #     end_conversation = False
    #     # If there is no repair request anymore
    #     if self.repair_request is None:
    #         end_conversation = True
    #     # And if the similarity of the inferred intention and the speaker's intention is maximised the conversation is
    #     # ended
    #     # TODO: finish function here according to communicate beliefs
    #
    #     return end_conversation

    def coherence(self, network):
        """
        Calculate coherence of a belief network.
        :return: float; coherence
        """

        # Initialise coherence
        coherence = 0

        for edge in list(network.edges(data='constraint')):
            if edge[2] == 'positive':
                if network.nodes[edge[0]]['truth_value'] == \
                        self.belief_network.nodes[edge[1]]['truth_value']:
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

if __name__ == '__main__':
    belief_network = CoherenceNetworks(10, 'low', 'middle').create_graph()
    SpeakerModel(belief_network, ['own', 'own', 'com', None, 'inf', 'own', 'own', 'com', None, 'inf'],
                  [True, False, True, True, False, True, True, False, True, False], [0, 1, 9, 3],
                 repair_request=[(0, False), (4, False), (5, False), (9, False)]).communicate_beliefs()
