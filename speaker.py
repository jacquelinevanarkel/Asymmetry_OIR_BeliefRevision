# Import packages
import networkx as nx
import random
from coherence_networks import CoherenceNetworks

class SpeakerModel:

    def __init__(self, belief_network, node_type, node_truth_value, repair_request=None, init=True):
        """
        Initialisation of class.
        :param belief_network: graph; the graph containing the nodes connected by edges with their constraints
        :param node_type: list; a list containing the types of all the nodes in the network (None if not specified)
        :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the network
        :param repair_request: list; list of tuples with the nodes (index, truth value) over which repair is asked
        """

        if init:
            self.belief_network = belief_network
            self.node_type = node_type
            self.node_truth_value = node_truth_value
            self.repair_request = repair_request

            # Initialise a similarity score to keep track of the similarity between the inferred intention and the
            # speaker's intention
            self.similarity = 0

            # Add the truth values and the type to the nodes in the belief network
            for i in range(len(node_truth_value)):
                self.belief_network.nodes[i]['truth_value'] = self.node_truth_value[i]
                self.belief_network.nodes[i]['type'] = self.node_type[i]

            # Initialise all nodes with communicated set to false, to keep track of which nodes are already communicated
            # by the speaker
            nx.set_node_attributes(self.belief_network, False, "communicated")

    def communicate_belief(self):
        """
        Communicate a (set of) node(s), which will be chosen such that when a belief revision is performed
        (with an egocentric listener model) it is most similar to the communicative intentions and the utterance should
        be as short as possible. Already communicated nodes can't be communicated again.
        :return: list; the truth values of a (set of) node(s) to be communicated (the other nodes will be set to 'None')
        """

        # TODO: finish communicate beliefs

        # https://docs.python.org/3/library/itertools.html --> look at subsets enumereren
        #########################@@@@@@@@@@@@@@@@@@ MW COMMENT #7: @@@@@@@@@@@@@@@@@@#########################
        # In response to your comment above about the itertools library: I think you're probably gonna want to
        # use the combinations() function from the itertools library, in combination with a for-loop that
        # iterates over all possible subset sizes, where you use that subset size as the second input argument
        # to the combinations() function (i.e. the "r" argument).
        #########################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#########################
        # --> for i in range(size): combinations(iteratable_object, i)

        # First of all, the speaker makes inferences about the nodes that are not its own beliefs or the communicative
        # intention (T'_inf)
        self.belief_revision()

        # This part should be done in a loop: for i in range(n_nodes) --> combinations(nodes, i) where nodes are nodes
        # that haven't been communicated yet
        # Calculate the similarity between the inferred intention and the speaker's intention
        # The speaker infers the intention by doing a belief revision with its own beliefs and a certain utterance

        # Calculate the 'optimisation': maximise the similarity divided by the utterance length

    def belief_revision(self):
        """
        Make inferences based on own and communicated beliefs (same function as for listener).
        :return: list; the truth values of a set of inferred nodes
        """

        print("Network before belief revision \n", self.belief_network.nodes(data=True))

        # Greedy algorithm: flip the node (inferred or None) that has the highest gain in coherence
        # Repeat for 2 times the number of nodes in the network
        for _ in range(2 * self.belief_network.number_of_nodes()):
            # Initialise a list that keeps track of the gain of coherence when the node would be flipped for all the
            # nodes (inferred or None) in the network
            gain_coherence = []

            # Check which node flip increases coherence most of the possible nodes in the network (inferred or None)
            for node in self.belief_network.nodes(data=True):

                # When the truth value is set to None, initialise it with a random truth value and set the type to
                # inferred
                if node[1]['type'] is None:
                    node[1]['truth_value'] = random.choice([True, False])
                    node[1]['type'] = 'inf'

                # Only flip inferred nodes
                if node[1]['type'] == 'inf':
                    # Calculate the coherence before the node is flipped
                    coherence = self.coherence()

                    # Calculate the coherence when the node is flipped and store in the list of coherence gain (together
                    # with the node index)
                    node[1]['truth_value'] = not node[1]['truth_value']
                    coherence_new = self.coherence()
                    gain_coherence.append((node[0], coherence_new - coherence))

                    # Flip the truth value of the node back to its original truth value
                    node[1]['truth_value'] = not node[1]['truth_value']

            # Check which flip of the truth value of a node has the highest gain in coherence and flip its truth value
            # when the gain in coherence is higher than 0

            # First the highest gain in coherence is found in the array containing tuples of all the gains of coherence
            # and their corresponding node indices
            node_flipped = max(gain_coherence, key=lambda x: x[1])

            # If the highest gain in coherence is bigger than 0, the corresponding node is flipped
            if node_flipped[1] > 0:
                print("Node belief revision: ", node_flipped[0], "gain: ", node_flipped[1])
                self.belief_network.nodes[node_flipped[0]]['truth_value'] = \
                    not self.belief_network.nodes[node_flipped[0]]['truth_value']

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

    def end_conversation(self):
        """
        End the conversation if no repair is initiated anymore and the speaker has communicated its intentions.
        :return: boolean; true if the conversation should be ended, false otherwise
        """

        end_conversation = False
        # If there is no repair request anymore
        if self.repair_request is None:
            end_conversation = True
        # And if the similarity of the inferred intention and the speaker's intention is maximised the conversation is
        # ended
        # TODO: finish function here according to communicate beliefs

        return end_conversation

    def coherence(self):
        """
        Calculate coherence of a belief network.
        :return: float; coherence
        """

        # Initialise coherence
        coherence = 0

        for edge in list(self.belief_network.edges(data='constraint')):
            if edge[2] == 'positive':
                if self.belief_network.nodes[edge[0]]['truth_value'] == \
                        self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence += 1
                else:
                    coherence -= 1
            elif edge[2] == 'negative':
                if self.belief_network.nodes[edge[0]]['truth_value'] == \
                        self.belief_network.nodes[edge[1]]['truth_value']:
                    coherence -= 1
                else:
                    coherence += 1

        # print("Coherence = ", coherence)

        return coherence

if __name__ == '__main__':
    belief_network = CoherenceNetworks(10, 'low', 'middle').create_graph()
    SpeakerModel(belief_network, ['own', 'own', 'com', None, 'inf', 'own', 'own', 'com', None, 'inf'],
                  [True, False, True, True, False, True, True, False, True, False]).repair_solution([(0, False), (4, False), (5, False), (9, False)])
