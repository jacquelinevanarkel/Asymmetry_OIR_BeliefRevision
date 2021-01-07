# Import packages and functions
from coherence_networks import CoherenceNetworks
from listener import ListenerModel
from speaker import SpeakerModel
import networkx as nx


def conversation(belief_network, node_type_listener, node_truth_value_listener, node_type_speaker,
                 node_truth_value_speaker):
    """
    Simulate one conversation between a speaker and listener, in which the speaker tries to communicate a certain
    intention to the listener and the listener can initiate repair if needed.
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief network for the listener and speaker
    :param node_type_listener: list; a list containing the types of all the nodes in the network (None if not specified)
    for the listener
    :param node_truth_value_listener: list; a list containing the truth values of (some of) the nodes in the listener's
    network
    :param node_type_speaker: list; a list containing the types of all the nodes in the network (None if not specified)
    for the speaker
    :param node_truth_value_speaker: list; a list containing the truth values of (some of) the nodes in the speaker's
    network
    :return: graph; complete belief networks of both the speaker and listener, including all its data (truth values,
    node types)
    """

    # A conversation can consist of a maximum of the number of nodes divided by 2 interactions
    for _ in range(belief_network.number_of_nodes()/2):
        # Speaker communicates something
        utterance, belief_network_speaker = SpeakerModel(belief_network, node_type_speaker, node_truth_value_speaker).\
            communicate_belief()

        # Listener changes beliefs accordingly and initiates repair if necessary
        repair_request, belief_network_listener = ListenerModel(belief_network, node_type_listener,
                                                                node_truth_value_listener,
                                                                communicated_nodes=utterance).belief_revision()

        # If the listener initiates repair the speaker gives a repair solution
        if repair_request:
            repair_solution = SpeakerModel(belief_network_speaker, node_type_speaker, node_truth_value_speaker,
                                           repair_request=repair_request, init=False).repair_solution()

            # The listener perform belief revision according to the repair solution from the speaker
            repair_request, belief_network_listener = ListenerModel(belief_network_listener, node_type_listener,
                                                                    node_truth_value_listener,
                                                                    communicated_nodes=repair_solution, init=False).\
                belief_revision()

        # If the listener does not initiate repair and the similarity is maximised the conversation is ended
        if not repair_request:
            maximum = 0
            for node in belief_network_speaker.nodes(data=True):
                if node[1]["type"] == 'intention':
                    maximum += 1
            if SpeakerModel(belief_network_speaker, node_type_speaker, node_truth_value_speaker).similarity == maximum:
                break

    # Return: asymmetry solved/intention correctly communicated, number of times repair is initiated, coherence score
    # per interaction, number of interactions per conversation, confirmation or disconfirmation in repair solution


def simulation(belief_network, node_type_listener, node_truth_value_listener, node_type_speaker,
               node_truth_value_speaker):
    """
    Multiple conversations for the same parameter settings and the same belief networks (structure-wise).
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief network for the listener and speaker
    :param node_type_listener: list; a list containing the types of all the nodes in the network (None if not specified)
    for the listener
    :param node_truth_value_listener: list; a list containing the truth values of (some of) the nodes in the listener's
    network
    :param node_type_speaker: list; a list containing the types of all the nodes in the network (None if not specified)
    for the speaker
    :param node_truth_value_speaker: list; a list containing the truth values of (some of) the nodes in the speaker's
    network
    """


def multi_runs(number_nodes, amount_edges, amount_positive_constraints):
    """
    Multiple simulations ran for different parameter settings and different belief networks (structure-wise).
    :param number_nodes: int; the number of nodes in the network
    :param amount_edges: string; the amount of edges connecting the nodes in the network (low, middle, high)
    :param amount_positive_constraints: string; the amount of positive constraints connecting the nodes in the
    network (low, middle, high)
    """

    belief_network = CoherenceNetworks(10, 'high', 'middle').create_graph()

    # Return: asymmetry solved/intention correctly communicated, number of times repair is initiated, coherence score
    # per interaction, number of interactions per conversation, confirmation or disconfirmation in repair solution
