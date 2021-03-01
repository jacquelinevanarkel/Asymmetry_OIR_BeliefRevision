# Import packages
import random
import itertools

class SpeakerModel:

    def __init__(self, belief_network, intention, repair_request=None):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the nodes connected by edges with their constraints
        :param repair_request: list; list of tuples with the nodes (index, truth value) over which repair is asked
        :param intention: list; list with the indices of the nodes that form the speaker's intention
        """

        self.belief_network = belief_network
        self.repair_request = repair_request
        self.intention = intention

    def communicate_beliefs(self):
        """
        Communicate a (set of) node(s), which will be chosen such that when a belief revision is performed
        (with an egocentric listener model) it is most similar to the communicative intentions and the utterance should
        be as short as possible. Already communicated nodes can't be communicated again.
        :return: list; the truth values of a (set of) node(s) to be communicated (the other nodes will be set to 'None')
        """

        # Get the not (yet) communicated nodes and its combinations of different sizes of (sub)sets
        not_comm_nodes = [x for x, y in self.belief_network.nodes(data=True) if y['type'] == 'inf' or
                          y['type'] == 'own']

        # If there are no nodes left to communicate, stop and indicate that nothing can be communicated
        if not not_comm_nodes:
            return [], self.belief_network, None

        # For every combination of (subsets) of nodes, calculate the similarity after belief revision and divide over
        # the number of nodes

        # Initialise a list for combinations of (sub)sets of nodes to be uttered
        combinations = []

        # Add all those (sub)sets to a list of combinations
        for r in range(1, len(not_comm_nodes)+1):
            combinations.extend(list(itertools.combinations(not_comm_nodes, r)))

        # Perform belief revision for every combination and calculate the similarity and divide over the number of nodes
        optimisation = []
        similarities = []
        for combination in combinations:
            network = self.belief_network.copy()
            network_listener = self.belief_revision(network, communicated_nodes=combination)

            similarity = 0
            # Calculate similarity
            for node in self.intention:
                if network_listener.nodes[node]['truth_value'] == self.belief_network.nodes[node]['truth_value']:
                    similarity += 1
            similarities.append(similarity)
            optimisation.append(similarity/len(combination))

        # The utterance is the combination with the highest optimisation
        max_optimisation = max(optimisation)
        max_indices = [i for i in range(len(optimisation)) if optimisation[i] == max_optimisation]
        optimisation_index = random.choice(max_indices)
        utterance_indices = combinations[optimisation_index]

        utterance = []
        for index in utterance_indices:
            utterance.append((index, self.belief_network.nodes[index]['truth_value']))
            self.belief_network.nodes[index]['type'] = 'com'

        return utterance, self.belief_network, similarities[optimisation_index]

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

        # Check whether the truth values of repair request match with the speaker's belief_network, if not, no
        # confirmation can be given
        confirmation = True
        for node in self.repair_request:
            if node[1] != self.belief_network.nodes[node[0]]['truth_value']:
                confirmation = False
                break

        # If no confirmation can be given, the speaker communicates their own truth values of the repair request and
        # gives an additional clarification
        repair = []
        clarification = []
        if not confirmation:
            for node in self.repair_request:
                repair.append((node[0], self.belief_network.nodes[node[0]]['truth_value']))
                self.belief_network.nodes[node[0]]['type'] = 'com'
            clarification, self.belief_network, similarity = self.communicate_beliefs()
            repair_solution = repair + clarification
        else:
            repair_solution = self.repair_request
            similarity = False

        return repair_solution, similarity, self.belief_network

    def coherence(self, network):
        """
        Calculate coherence of a belief belief_network.
        :return: float; coherence
        """

        # Initialise coherence
        coherence = 0

        for edge in list(network.edges(data='constraint')):
            if edge[2] == 'positive':
                if network.nodes[edge[0]]['truth_value'] == \
                        network.nodes[edge[1]]['truth_value']:
                    coherence += 1
                else:
                    coherence -= 1
            elif edge[2] == 'negative':
                if network.nodes[edge[0]]['truth_value'] == \
                        network.nodes[edge[1]]['truth_value']:
                    coherence -= 1
                else:
                    coherence += 1

        return coherence
