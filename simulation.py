# Import packages and functions
from coherence_networks import CoherenceNetworks
from listener import ListenerModel
from speaker import SpeakerModel
import networkx as nx
import random


def conversation(belief_network, node_type_listener, node_truth_value_listener, node_type_speaker,
                 node_truth_value_speaker, intention):
    """
    Simulate one conversation between a speaker and listener, in which the speaker tries to communicate a certain
    intention to the listener and the listener can initiate repair if needed.
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief belief_network for the listener and speaker
    :param node_type_listener: list; a list containing the types of all the nodes in the belief_network (None if not specified)
    for the listener
    :param node_truth_value_listener: list; a list containing the truth values of (some of) the nodes in the listener's
    belief_network
    :param node_type_speaker: list; a list containing the types of all the nodes in the belief_network (None if not specified)
    for the speaker
    :param node_truth_value_speaker: list; a list containing the truth values of (some of) the nodes in the speaker's
    belief_network
    :param intention: list; list with the indices of the nodes that form the speaker's intention
    :return: graph; complete belief networks of both the speaker and listener, including all its data (truth values,
    node types)
    """

    # Initialisation of speaker and listener belief_network
    belief_network_speaker = initialisation_network(belief_network, node_type_speaker, node_truth_value_speaker,
                                                    "speaker")
    belief_network_listener = initialisation_network(belief_network.copy(), node_type_listener, node_truth_value_listener,
                                                     "listener")

    print("Speaker belief_network: \n", belief_network_speaker.nodes(data=True))
    print("Listener belief_network: \n", belief_network_listener.nodes(data=True))

    # A conversation can consist of a maximum of the number of nodes interactions
    for _ in range(belief_network.number_of_nodes()):

        # Speaker communicates something
        utterance, belief_network_speaker, similarity = SpeakerModel(belief_network_speaker, intention).communicate_beliefs()
        print("Speaker belief_network: \n", belief_network_speaker.nodes(data=True))
        print("Speaker communicates: ", utterance)

        # Stop if the speaker has nothing left to say
        if utterance is None:
            break

        # Listener changes beliefs accordingly and initiates repair if necessary
        repair_request, belief_network_listener = ListenerModel(belief_network_listener, communicated_nodes=utterance)\
            .belief_revision()
        print("Listener belief_network: \n", belief_network_listener.nodes(data=True))
        print("Repair request: ", repair_request)

        # If the listener initiates repair the speaker gives a repair solution
        if repair_request:
            repair_solution, similarity = SpeakerModel(belief_network_speaker, intention, repair_request=repair_request)\
                .repair_solution()
            print("Repair solution: ", repair_solution)

            # The listener performs belief revision according to the repair solution from the speaker
            repair_request, belief_network_listener = ListenerModel(belief_network_listener,
                                                                    communicated_nodes=repair_solution)\
                .belief_revision()
            print("Listener belief_network: \n", belief_network_listener.nodes(data=True))
            print("Repair request after repair solution: ", repair_request)

        # If the listener does not initiate repair and the similarity is maximised the conversation is ended
        if not repair_request:
            maximum = len(intention)
            if similarity == maximum:
                print("Conversation ended because maximum similarity is achieved")
                break

    # Return: asymmetry solved/intention correctly communicated, number of times repair is initiated, coherence score
    # per interaction, number of interactions per conversation, confirmation or disconfirmation in repair solution


def initialisation_network(belief_network, node_type, node_truth_value, agent):
    """
    Initialisation of belief belief_network with the starting node types and truth values for the nodes.
    :param agent:
    :param belief_network:
    :param node_type:
    :param node_truth_value:
    :return:
    """

    # If a type value is none, set it to inferred so they can be inferred in belief_revision()
    node_type = ['inf' if type is None else type for type in node_type]

    # Add the truth values and the type to the nodes in the belief belief_network
    for i in range(len(node_truth_value)):
        belief_network.nodes[i]['truth_value'] = node_truth_value[i]
        belief_network.nodes[i]['type'] = node_type[i]

    # If the agent is a listener a node attribute should be initialised whether repair was already asked over that node
    if agent == "listener":
        # Initialise all the nodes of the listener with repair set to false as repair has not been used yet
        nx.set_node_attributes(belief_network, False, "repair")

    return belief_network


def simulation(belief_network, degree_overlap, degree_asymmetry):
    """
    Multiple conversations for the same parameter settings and the same belief networks (structure-wise).
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief belief_network for the listener and speaker
    :param node_type_listener: list; a list containing the types of all the nodes in the belief_network (None if not specified)
    for the listener
    :param node_truth_value_listener: list; a list containing the truth values of (some of) the nodes in the listener's
    belief_network
    :param node_type_speaker: list; a list containing the types of all the nodes in the belief_network (None if not specified)
    for the speaker
    :param node_truth_value_speaker: list; a list containing the truth values of (some of) the nodes in the speaker's
    belief_network
    """

    # Two manipulations: degree of overlap of the own belief sets and degree of asymmetry within these overlapping sets

    # First the degree of overlap between the sets of own nodes is set
    n_nodes = belief_network.number_of_nodes()
    if degree_overlap == 100:
        node_type_speaker = random.choices(["own", "inf"], k=n_nodes)
        node_type_listener = node_type_speaker
    elif degree_overlap == 50:
        # Maximum of 60% can be own beliefs: first randomly choose a percentage under 60 and then randomly choose the
        # corresponding amount of indices to put 'own' in
        percentage = random.randint(1, 60)
        indices_speaker_own = random.sample(list(range(n_nodes)), k=int((percentage/100)*n_nodes))
        print("speaker_own: ", indices_speaker_own)
        node_type_speaker = ["own" if n in indices_speaker_own else "inf" for n in range(n_nodes)]
        indices_own = random.sample(indices_speaker_own, k=int(0.5*len(indices_speaker_own)))
        print("indices own:", indices_own)
        node_type_listener = ["own" if n in indices_own else None for n in range(n_nodes)]
        print("node_type listener:", node_type_listener)
        indices_inf = [i for i in range(len(node_type_speaker)) if node_type_speaker[i] == "inf"]
        print("indices inf", indices_inf)
        node_type_listener = ["inf" if node_type_listener.index(n) in indices_inf else n for n in node_type_listener]
        print("node type listener", node_type_listener)
        n_own = node_type_speaker.count("own")
        n_own_left = n_own - len(indices_own)
        indices_own_left = [node_type_listener.index(node) for node in node_type_listener if node is None]
        print("indices own left", indices_own_left)
        indices_own = random.sample(indices_own_left, k=n_own_left)
        print(indices_own)
        node_type_listener = ["own" if node_type_listener.index(n) in indices_own else n for n in node_type_listener]
    elif degree_overlap == 0:
        node_type_speaker = random.choices(["own", "inf"], k=n_nodes)
        indices_speaker_inf = [i for i in range(len(node_type_speaker)) if node_type_speaker[i] == "inf"]
        n_own_speaker = len(node_type_speaker) - len(indices_speaker_inf)
        indices_own_listener = random.sample(indices_speaker_inf, k=n_own_speaker)
        node_type_listener = ["own" if i in indices_own_listener else "inf" for i in range(len(node_type_speaker))]

    print("Node types speaker: ", node_type_speaker)
    print("Node types listener: ", node_type_listener)


    truth_value_speaker = None
    truth_value_listener = None


    # The truth values for the belief belief_network are generated independently from each other and only the overlapping parts
    # of the own beliefs will be made the same and changed accordingly to the degree of asymmetry. This will be done by
    # flipping some truth values randomly, divided over the two belief networks for the speaker and listener.


def multi_runs(number_nodes, amount_edges, amount_positive_constraints):
    """
    Multiple simulations ran for different parameter settings and different belief networks (structure-wise).
    :param number_nodes: int; the number of nodes in the belief_network
    :param amount_edges: string; the amount of edges connecting the nodes in the belief_network (low, middle, high)
    :param amount_positive_constraints: string; the amount of positive constraints connecting the nodes in the
    belief_network (low, middle, high)
    """

    belief_network = CoherenceNetworks(10, 'high', 'middle').create_graph()

    # Return: networks (speaker and listener) of every turn with all edge and node dataa, number of times repair is
    # initiated, coherence score per turn, number of interactions per conversation (keep track when a conversation
    # starts/ends as you already store the networks or a simple count?), confirmation or disconfirmation in repair
    # solution

if __name__ == '__main__':
    belief_network = CoherenceNetworks(10, 'middle', 'middle').create_graph()
    # conversation(belief_network, ['own', 'own', None, None, 'inf', 'own', 'own', None, None, 'inf'],
    #              [False, True, True, True, False, False, True, True, True, False],
    #              ['own', 'own', None, None, 'inf', 'own', 'own', None, None, 'inf'],
    #              [True, False, True, True, False, True, True, False, True, False], [0, 1, 9, 3])
    simulation(belief_network, 50, 0)
