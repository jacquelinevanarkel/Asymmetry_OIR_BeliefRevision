# Import packages and functions
from coherence_networks import CoherenceNetworks
from listener import ListenerModel
from speaker import SpeakerModel
import networkx as nx
import random
import pandas as pd
import multiprocessing
import pickle


def conversation(belief_network_speaker, belief_network_listener, intention):
    """
    Simulate one conversation between a speaker and listener, in which the speaker tries to communicate a certain
    intention to the listener and the listener can initiate repair if needed.
    :param belief_network_speaker: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the speaker
    :param belief_network_listener: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the listener
    :param intention: list; list with the indices of the nodes that form the speaker's intention
    :return: dataframe; pandas dataframe containing the results of the conversation
    """

    # Initialise a dataframe to store the results in
    results = pd.DataFrame(
        columns=["nodes speaker", "nodes listener", "edges speaker", "edges listener", "intention_communicated",
                 "n_repair", "coherence speaker", "coherence listener", "n_interactions", "confirmation?",
                 "conversation state", "similarity", "utterance speaker", "repair request",
                 "conversation ended max sim", "intention", "asymmetry", "n_turns"])

    # print("Speaker belief_network: \n", belief_network_speaker.nodes(data=True))
    # print("Listener belief_network: \n", belief_network_listener.nodes(data=True))

    # Initialise the listener belief network in order to make inferences before the speaker communicates something
    # Listener changes beliefs accordingly and initiates repair if necessary
    repair_request, belief_network_listener = ListenerModel(belief_network_listener.copy()).belief_revision()

    # Initialise a count for the number of turns taken in a conversation
    t = 0

    # Store the starting conditions in the results
    results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                 belief_network_listener.nodes(data=True),
                                 belief_network_speaker.edges(data=True),
                                 belief_network_listener.edges(data=True), None, None,
                                 coherence(belief_network_speaker),
                                 coherence(belief_network_listener),
                                 None, None, "Start", None, None, None, False, intention,
                                 asymmetry_count(belief_network_speaker, belief_network_listener), t]

    # A conversation can consist of a maximum of the number of nodes interactions
    # Initialise a count for the number of times repair is initiated in a conversation
    r = 0
    for i in range(belief_network_speaker.number_of_nodes()):

        # Speaker communicates something
        utterance, belief_network_speaker, similarity = SpeakerModel(belief_network_speaker.copy(),
                                                                     intention).communicate_beliefs()
        t += 1
        # print("Speaker belief_network: \n", belief_network_speaker.nodes(data=True))
        # print("Speaker communicates: ", utterance)

        # Store results
        results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                     belief_network_listener.nodes(data=True),
                                     belief_network_speaker.edges(data=True),
                                     belief_network_listener.edges(data=True), None, None,
                                     coherence(belief_network_speaker),
                                     coherence(belief_network_listener),
                                     None, None, i, similarity, utterance, None, False, intention,
                                     asymmetry_count(belief_network_speaker, belief_network_listener), t]

        # Stop if the speaker has nothing left to say
        if not utterance:
            break

        # Listener changes beliefs accordingly and initiates repair if necessary
        repair_request, belief_network_listener = ListenerModel(belief_network_listener.copy(),
                                                                communicated_nodes=utterance).belief_revision()
        if repair_request:
            t += 1
        # print("Listener belief_network: \n", belief_network_listener.nodes(data=True))
        # print("Repair request: ", repair_request)

        # Store results
        results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                     belief_network_listener.nodes(data=True),
                                     belief_network_speaker.edges(data=True),
                                     belief_network_listener.edges(data=True), None, None,
                                     coherence(belief_network_speaker),
                                     coherence(belief_network_listener),
                                     None, None, i, None, None, repair_request, False, intention,
                                     asymmetry_count(belief_network_speaker, belief_network_listener), t]

        # If the listener initiates repair the speaker gives a repair solution
        if repair_request:
            r += 1
            repair_solution, similarity = SpeakerModel(belief_network_speaker.copy(), intention,
                                                       repair_request=repair_request).repair_solution()
            t += 1
            # print("Repair solution: ", repair_solution)

            confirmation = False
            if repair_solution == repair_request:
                confirmation = True

            # Store results
            results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                         belief_network_listener.nodes(data=True),
                                         belief_network_speaker.edges(data=True),
                                         belief_network_listener.edges(data=True), None, None,
                                         coherence(belief_network_speaker),
                                         coherence(belief_network_listener),
                                         None, confirmation, i, similarity, repair_solution,
                                         repair_request, False, intention,
                                         asymmetry_count(belief_network_speaker, belief_network_listener), t]

            # The listener performs belief revision according to the repair solution from the speaker
            repair_request, belief_network_listener = ListenerModel(belief_network_listener.copy(),
                                                                    communicated_nodes=repair_solution) \
                .belief_revision()
            # print("Listener belief_network: \n", belief_network_listener.nodes(data=True))
            # print("Repair request after repair solution: ", repair_request)

            # Store results
            results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                         belief_network_listener.nodes(data=True),
                                         belief_network_speaker.edges(data=True),
                                         belief_network_listener.edges(data=True), None, None,
                                         coherence(belief_network_speaker),
                                         coherence(belief_network_listener),
                                         None, None, i, None, None,
                                         repair_request, False, intention,
                                         asymmetry_count(belief_network_speaker, belief_network_listener), t]

        # If the listener does not initiate repair and the similarity is maximised the conversation is ended
        max_sim_end = False
        if not repair_request:
            maximum = len(intention)
            if similarity == maximum:
                max_sim_end = True
                # Store results
                results.loc[len(results)] = [belief_network_speaker.nodes(data=True),
                                             belief_network_listener.nodes(data=True),
                                             belief_network_speaker.edges(data=True),
                                             belief_network_listener.edges(data=True), None,
                                             None,
                                             coherence(belief_network_speaker),
                                             coherence(belief_network_listener),
                                             None, None, i, similarity, None,
                                             repair_request, max_sim_end, intention,
                                             asymmetry_count(belief_network_speaker, belief_network_listener), t]
                break

    # Add conversation info to results
    results.intention_communicated = intention_communicated(belief_network_speaker, belief_network_listener, intention)
    results.n_repair = r
    results.n_interactions = i+1

    return results


def intention_communicated(belief_network_speaker, belief_network_listener, intention):
    """
    Returns whether the speaker's intention matches with the listener's nodes.
    :param belief_network_speaker: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the speaker
    :param belief_network_listener: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the listener
    :param intention: list; list with the indices of the nodes that form the speaker's intention
    :return: boolean; whether the intention matches with the corresponding listener's nodes
    """
    for index in intention:
        if belief_network_speaker.nodes[index]['truth_value'] != belief_network_listener.nodes[index]['truth_value']:
            return False

    return True

def asymmetry_count(belief_network_speaker, belief_network_listener):
    """
    Counts the asymmetry between the belief network of the speaker and listener.
    :param belief_network_speaker: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the speaker
    :param belief_network_listener: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the listener
    :return: int; the asymmetry of the networks
    """

    # Initialise a count for the asymmetry
    i = 0
    for index in range(belief_network_speaker.number_of_nodes()):
        if belief_network_speaker.nodes[index]['truth_value'] != belief_network_listener.nodes[index]['truth_value']:
            i += 1

    return i

def coherence(network):
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


def simulation():
    """
    Multiple conversations for the same parameter settings and the same belief networks (structure-wise).
    """

    # Initialise a dataframe to store the results in
    results = pd.DataFrame(
        columns=["nodes speaker", "nodes listener", "edges speaker", "edges listener", "intention_communicated",
                 "n_repair", "coherence speaker", "coherence listener", "n_interactions", "confirmation?",
                 "conversation state", "similarity", "utterance speaker", "repair request",
                 "conversation ended max sim", "intention", "asymmetry", "n_turns"])

    # Initialise empty list to store the different arguments in
    speaker_network = []
    listener_network = []
    intentions = []
    list_n_nodes = []
    list_amount_edges = []
    list_amount_positive_constraints = []
    list_degree_overlap = []
    list_degree_asymmetry = []

    for _ in range(1):
        # Initialisation of the belief networks for the speaker and listener

        # First the possible combinations of amount of nodes, edges and positive constraints are used to generate a
        # network
        n_nodes = [8, 10, 12]
        for a in n_nodes:
            number_nodes = a
            amount = ["middle", "high"]
            for x in amount:
                amount_edges = x
                amount_positive_constraints = "middle"
                belief_network = CoherenceNetworks(number_nodes, amount_edges, amount_positive_constraints).\
                    create_graph()

                # Then the possible combinations of the degree of overlap and asymmetry are used to initialise the
                # network for the speaker and listener
                degree = [0, 50, 100]
                for i in degree:
                    degree_overlap = i
                    for n in degree:
                        degree_asymmetry = n
                        belief_network_speaker, belief_network_listener = initialisation_networks(belief_network,
                                                                                                  degree_overlap,
                                                                                                  degree_asymmetry)
                        # Randomly generate an intention for the speaker
                        n_nodes_intention = random.randint(int(0.25*number_nodes), int(0.75*number_nodes))
                        intention = random.sample(list(range(number_nodes)), k=n_nodes_intention)

                        # Collect arguments
                        speaker_network.append(belief_network_speaker)
                        listener_network.append(belief_network_listener)
                        intentions.append(intention)

                        # Collect manipulations to store in dataframe
                        list_n_nodes.append(number_nodes)
                        list_amount_edges.append(amount_edges)
                        list_amount_positive_constraints.append(amount_positive_constraints)
                        list_degree_overlap.append(degree_overlap)
                        list_degree_asymmetry.append(degree_asymmetry)


    # Run a conversation for the specified parameter settings
    pool = multiprocessing.Pool()
    arguments = zip(speaker_network, listener_network, intentions)
    arg_list = list(arguments)
    result = pool.starmap(conversation, arg_list)
    for index in range(len(intentions)):
        result[index]["n_nodes"] = [list_n_nodes[index]] * len(result[index])
        result[index]["amount_edges"] = [list_amount_edges[index]] * len(result[index])
        result[index]["amount_pos_constraint"] = [list_amount_positive_constraints[index]] * len(result[index])
        result[index]["overlap"] = [list_degree_overlap[index]] * len(result[index])
        result[index]["asymmetry"] = [list_degree_asymmetry[index]] * len(result[index])
        result[index]["simulation_number"] = [index] * len(result[index])
        result[index]["ended max sim"] = [result[index].loc[result[index].index[-1], "conversation ended max sim"]] * len(result[index])
        results = results.append(result[index])
    pool.close()
    pool.join()

    # Pickle the results
    filename = "results.p"
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()


def initialisation_networks(belief_network, degree_overlap, degree_asymmetry):
    """
    Initialise the belief networks for the speaker and listener according to the degree of overlap and asymmetry.
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints as a
    belief network for the speaker and listener
    :param degree_overlap: int; either 0 (no overlap), 50 (50% overlap), or 100 (complete overlap) of the own beliefs of
    the speaker and listener
    :param degree_asymmetry: int; either 0 (no asymmetry), 50 (50% asymmetry), or 100 (100% asymmetry) of the
    overlapping own beliefs of the speaker and listener
    :return: graph; the belief networks for the speaker and listener
    """

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- Node types ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # First the degree of overlap between the sets of own nodes is set
    n_nodes = belief_network.number_of_nodes()
    if degree_overlap == 100:
        # Choose randomly for the speaker which nodes are own beliefs and inferred beliefs
        node_type_speaker = random.choices(["own", "inf"], k=n_nodes)
        # Copy the types for the listener as the overlap is 100%
        node_type_listener = node_type_speaker
    elif degree_overlap == 50:
        # Maximum of 60% can be own beliefs: first randomly choose a percentage under 60 and then randomly choose the
        # corresponding amount of indices to put 'own' in
        percentage = random.randint(1, 60)
        k = int((percentage / 100) * n_nodes)

        # You need at least two own beliefs in order to have 50% overlap
        if k < 2:
            k = 2

        # Here the indices for the speaker's own beliefs are randomly chosen
        indices_speaker_own = random.sample(list(range(n_nodes)), k=k)

        # Construct the speaker's list of node types
        node_type_speaker = ["own" if n in indices_speaker_own else "inf" for n in range(n_nodes)]

        # Take half of the indices of the speaker's own beliefs for the speaker
        indices_own_shared = random.sample(indices_speaker_own, k=int(0.5 * len(indices_speaker_own)))

        # Set these own beliefs in the listener's list of node types
        node_type_listener = ["own" if n in indices_own_shared else None for n in range(n_nodes)]

        # For the other speaker's own beliefs the listener needs to have inferred beliefs
        for index in indices_speaker_own:
            if index not in indices_own_shared:
                node_type_listener[index] = "inf"

        # The own belief set needs to be equal in size for speaker and listener
        # Count how many own beliefs the speaker has
        n_own = node_type_speaker.count("own")

        # Count how many own beliefs you still need to set for your listener
        n_own_left = n_own - len(indices_own_shared)

        # See which indices are left for the listener to set the own beliefs for
        indices_own_left = [i for i in range(len(node_type_listener)) if node_type_listener[i] is None]

        # Randomly choose which nodes will be the left over own beliefs for the listener
        indices_own = random.sample(indices_own_left, k=n_own_left)

        # And set the last own beliefs of the listener
        for index in indices_own:
            node_type_listener[index] = "own"
    elif degree_overlap == 0:
        # A maximum of half of the beliefs can be own in order not to have any overlap
        percentage = random.randint(1, 50)
        k = int((percentage / 100) * n_nodes)

        # Here the indices for the speaker's own beliefs are randomly chosen
        indices_speaker_own = random.sample(list(range(n_nodes)), k=k)

        # Construct the speaker's list of node types
        node_type_speaker = ["own" if n in indices_speaker_own else "inf" for n in range(n_nodes)]

        # Get the indices for the speaker's inferred node types
        indices_speaker_inf = [i for i in range(len(node_type_speaker)) if node_type_speaker[i] == "inf"]

        # Get the number of own beliefs for the listener according to the speaker
        n_own_speaker = len(node_type_speaker) - len(indices_speaker_inf)

        # Randomly choose which nodes will be the own beliefs for the listener (i.e., where the speaker has inferred
        # nodes)
        indices_own_listener = random.sample(indices_speaker_inf, k=n_own_speaker)

        # Set the node types for the listener
        node_type_listener = ["own" if i in indices_own_listener else "inf" for i in range(len(node_type_speaker))]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- Truth values ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Independently generate the truth values for the nodes for the speaker and listener
    truth_value_speaker = random.choices([True, False], k=n_nodes)
    truth_value_listener = random.choices([True, False], k=n_nodes)

    if degree_overlap == 100:
        # Make the truth values of the overlapping own beliefs the same
        indices_speaker_own = [i for i in range(len(node_type_speaker)) if node_type_speaker[i] == "own"]
        for index in indices_speaker_own:
            if truth_value_speaker[index] != truth_value_listener[index]:
                truth_value_listener[index] = truth_value_speaker[index]
        # If the degree of asymmetry is 100, change all the listener's overlapping own truth values to the opposite
        # of the speaker's
        if degree_asymmetry == 100:
            for index in indices_speaker_own:
                truth_value_listener[index] = not truth_value_speaker[index]
        # If the degree of asymmetry is 50, change half of the listener's overlapping own truth values to the opposite
        # of the speaker's
        if degree_asymmetry == 50:
            k = int(len(indices_speaker_own) / 2)
            flip_indices = random.sample(indices_speaker_own, k=k)
            for index in flip_indices:
                truth_value_listener[index] = not truth_value_speaker[index]
    if degree_overlap == 50:
        # Make the truth values of the overlapping own beliefs the same
        for index in indices_own_shared:
            if truth_value_speaker[index] != truth_value_listener[index]:
                truth_value_listener[index] = truth_value_speaker[index]
        # If the degree of asymmetry is 100, change all the listener's overlapping own truth values to the opposite
        # of the speaker's
        if degree_asymmetry == 100:
            for index in indices_own_shared:
                truth_value_listener[index] = not truth_value_speaker[index]
        # If the degree of asymmetry is 50, change half of the listener's overlapping own truth values to the opposite
        # of the speaker's
        if degree_asymmetry == 50:
            k = int(len(indices_own_shared) / 2)
            flip_indices = random.sample(indices_own_shared, k=k)
            for index in flip_indices:
                truth_value_listener[index] = not truth_value_speaker[index]

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Initialisation ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Initialisation of speaker and listener belief_network
    belief_network_speaker = initialisation_network(belief_network, node_type_speaker, truth_value_speaker,
                                                    "speaker")
    belief_network_listener = initialisation_network(belief_network.copy(), node_type_listener,
                                                     truth_value_listener,
                                                     "listener")
    return belief_network_speaker, belief_network_listener


def initialisation_network(belief_network, node_type, node_truth_value, agent):
    """
    Initialisation of belief belief_network with the starting node types and truth values for the nodes.
    :param agent: string; the agent type for the network
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief network for the listener and speaker
    :param node_type: list; a list containing the types of all the nodes in the belief_network (None if not specified)
    :param node_truth_value: list; a list containing the truth values of (some of) the nodes in the belief_network
    :return: graph; the belief network with its node types and truth values
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

if __name__ == '__main__':
    simulation()