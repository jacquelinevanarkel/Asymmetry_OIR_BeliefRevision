# Import packages and functions
from coherence_networks import CoherenceNetworks
from interpreter import InterpreterModel
from producer import ProducerModel
import networkx as nx
import random
import pandas as pd
import multiprocessing
import pickle


def conversation(belief_network_producer, belief_network_interpreter, intention):
    """
    Simulate one conversation between a producer and interpreter, in which the producer tries to communicate a certain
    intention to the interpreter and the interpreter can initiate repair if needed.
    :param belief_network_producer: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the producer
    :param belief_network_interpreter: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the interpreter
    :param intention: list; list with the indices of the nodes that form the producer's intention
    :return: dataframe; pandas dataframe containing the results of the conversation
    """

    # Initialise a dataframe to store the results in
    results = pd.DataFrame(
        columns=["nodes producer", "nodes interpreter", "edges producer", "edges interpreter", "intention_communicated",
                 "n_repair", "coherence producer", "coherence interpreter", "n_interactions", "confirmation?",
                 "conversation state", "similarity", "utterance producer", "repair request",
                 "conversation ended max sim", "intention", "asymmetry_count", "n_turns", "asymmetry_intention", "state"])

    # print("producer belief_network: \n", belief_network_producer.nodes(data=True))
    # print("interpreter belief_network: \n", belief_network_interpreter.nodes(data=True))

    # Initialise the interpreter belief network in order to make inferences before the producer communicates something
    # interpreter changes beliefs accordingly
    repair_request, belief_network_interpreter = InterpreterModel(belief_network_interpreter.copy()).belief_revision()

    # Initialise the producer belief network
    belief_network_producer = ProducerModel(belief_network_producer.copy(), intention).\
        belief_revision(belief_network_producer.copy())

    # Initialise a count for the number of turns taken in a conversation
    t = 0

    # Store the starting conditions in the results
    results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                 belief_network_interpreter.nodes(data=True),
                                 belief_network_producer.edges(data=True),
                                 belief_network_interpreter.edges(data=True), None, None,
                                 coherence(belief_network_producer),
                                 coherence(belief_network_interpreter),
                                 None, None, "Start", None, None, None, False, intention,
                                 asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                 asymmetry_count(belief_network_producer, belief_network_interpreter, intention=intention),
                                 "interpreter initialisation"]

    # A conversation can consist of a maximum of the number of nodes interactions
    # Initialise a count for the number of times repair is initiated in a conversation
    r = 0
    for i in range(belief_network_producer.number_of_nodes()):

        # producer communicates something
        utterance, belief_network_producer, similarity = ProducerModel(belief_network_producer.copy(),
                                                                       intention).communicate_beliefs()
        t += 1
        # print("producer belief_network: \n", belief_network_producer.nodes(data=True))
        # print("producer communicates: ", utterance)

        # Store results
        results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                     belief_network_interpreter.nodes(data=True),
                                     belief_network_producer.edges(data=True),
                                     belief_network_interpreter.edges(data=True), None, None,
                                     coherence(belief_network_producer),
                                     coherence(belief_network_interpreter),
                                     None, None, i, similarity, utterance, None, False, intention,
                                     asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                     asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                     intention=intention), "initialisation"]

        # Stop if the producer has nothing left to say
        if not utterance:
            break

        # interpreter changes beliefs accordingly and initiates repair if necessary
        repair_request, belief_network_interpreter = InterpreterModel(belief_network_interpreter.copy(),
                                                                      communicated_nodes=utterance).belief_revision()

        # print("interpreter belief_network: \n", belief_network_interpreter.nodes(data=True))
        # print("Repair request: ", repair_request)

        # Store results
        results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                     belief_network_interpreter.nodes(data=True),
                                     belief_network_producer.edges(data=True),
                                     belief_network_interpreter.edges(data=True), None, None,
                                     coherence(belief_network_producer),
                                     coherence(belief_network_interpreter),
                                     None, None, i, None, None, repair_request, False, intention,
                                     asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                     asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                     intention=intention), "interpreter update utterance"]

        # If the interpreter initiates repair the producer gives a repair solution
        clarification = -1
        while repair_request:
            clarification += 1
            r += 1
            t += 1
            repair_solution, similarity, belief_network_producer = ProducerModel(belief_network_producer.copy(), intention,
                                                                                 repair_request=repair_request)\
                .repair_solution()
            t += 1
            # print("Repair solution: ", repair_solution)

            confirmation = False
            if repair_solution == repair_request:
                confirmation = True

            # Store results

            if clarification == 0:
                results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                             belief_network_interpreter.nodes(data=True),
                                             belief_network_producer.edges(data=True),
                                             belief_network_interpreter.edges(data=True), None, None,
                                             coherence(belief_network_producer),
                                             coherence(belief_network_interpreter),
                                             None, confirmation, i, similarity, repair_solution,
                                             repair_request, False, intention,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                             intention=intention), "producer update network repair"]

                # The interpreter performs belief revision according to the repair solution from the producer
                repair_request, belief_network_interpreter = InterpreterModel(belief_network_interpreter.copy(),
                                                                              communicated_nodes=repair_solution) \
                    .belief_revision()
                # print("interpreter belief_network: \n", belief_network_interpreter.nodes(data=True))
                # print("Repair request after repair solution: ", repair_request)

                # Store results
                results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                             belief_network_interpreter.nodes(data=True),
                                             belief_network_producer.edges(data=True),
                                             belief_network_interpreter.edges(data=True), None, None,
                                             coherence(belief_network_producer),
                                             coherence(belief_network_interpreter),
                                             None, None, i, None, None,
                                             repair_request, False, intention,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                             intention=intention), "interpreter update solution"]
            else:
                results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                             belief_network_interpreter.nodes(data=True),
                                             belief_network_producer.edges(data=True),
                                             belief_network_interpreter.edges(data=True), None, None,
                                             coherence(belief_network_producer),
                                             coherence(belief_network_interpreter),
                                             None, confirmation, i, similarity, repair_solution,
                                             repair_request, False, intention,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                             intention=intention), "producer update network repair_" + str(clarification)]

                # The interpreter performs belief revision according to the repair solution from the producer
                repair_request, belief_network_interpreter = InterpreterModel(belief_network_interpreter.copy(),
                                                                              communicated_nodes=repair_solution) \
                    .belief_revision()
                # print("interpreter belief_network: \n", belief_network_interpreter.nodes(data=True))
                # print("Repair request after repair solution: ", repair_request)

                # Store results
                results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                             belief_network_interpreter.nodes(data=True),
                                             belief_network_producer.edges(data=True),
                                             belief_network_interpreter.edges(data=True), None, None,
                                             coherence(belief_network_producer),
                                             coherence(belief_network_interpreter),
                                             None, None, i, None, None,
                                             repair_request, False, intention,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                             intention=intention), "interpreter update solution_" + str(clarification)]

        # If the interpreter does not initiate repair and the similarity is maximised the conversation is ended
        max_sim_end = False
        if not repair_request:
            maximum = len(intention)
            if similarity == maximum:
                max_sim_end = True
                # Store results
                results.loc[len(results)] = [belief_network_producer.nodes(data=True),
                                             belief_network_interpreter.nodes(data=True),
                                             belief_network_producer.edges(data=True),
                                             belief_network_interpreter.edges(data=True), None,
                                             None,
                                             coherence(belief_network_producer),
                                             coherence(belief_network_interpreter),
                                             None, None, i, similarity, None,
                                             repair_request, max_sim_end, intention,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter), t,
                                             asymmetry_count(belief_network_producer, belief_network_interpreter,
                                                             intention=intention), "end conversation"]
                break

    # Add conversation info to results
    results.intention_communicated = intention_communicated(belief_network_producer, belief_network_interpreter, intention)
    results.n_repair = r
    results.n_interactions = i + 1

    return results


def intention_communicated(belief_network_producer, belief_network_interpreter, intention):
    """
    Returns whether the producer's intention matches with the interpreter's nodes.
    :param belief_network_producer: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the producer
    :param belief_network_interpreter: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the interpreter
    :param intention: list; list with the indices of the nodes that form the producer's intention
    :return: boolean; whether the intention matches with the corresponding interpreter's nodes
    """

    for index in intention:
        if belief_network_producer.nodes[index]['truth_value'] != belief_network_interpreter.nodes[index]['truth_value']:
            return False

    return True


def asymmetry_count(belief_network_producer, belief_network_interpreter, intention=None):
    """
    Counts the asymmetry using the Hamming distance between the belief network of the producer and interpreter.
    :param belief_network_producer: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the producer
    :param belief_network_interpreter: graph; the graph containing the relevant nodes (including their truth values and
    types) connected by edges with their constraints as a belief belief_network for the interpreter
    :param intention: list; the list containing the node indices of the intention of the producer
    :return: int; the asymmetry of (a part of) the networks
    """

    # Initialise a count for the asymmetry
    i = 0
    if not intention:
        for index in range(belief_network_producer.number_of_nodes()):
            if belief_network_producer.nodes[index]['truth_value'] != belief_network_interpreter.nodes[index]['truth_value']:
                i += 1
    else:
        for index in intention:
            if belief_network_producer.nodes[index]['truth_value'] != belief_network_interpreter.nodes[index]['truth_value']:
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


def simulation(n_nodes, n_runs):
    """
    Multiple conversations for the same parameter settings and the same belief networks (structure-wise).
    :param n_nodes: int; the number of nodes to be used for the simulation
    :param n_runs: int; the number of runs for every simulation (combination of parameter settings)
    """

    # Initialise a dataframe to store the results in
    results = pd.DataFrame(
        columns=["nodes producer", "nodes interpreter", "edges producer", "edges interpreter", "intention_communicated",
                 "n_repair", "coherence producer", "coherence interpreter", "n_interactions", "confirmation?",
                 "conversation state", "similarity", "utterance producer", "repair request",
                 "conversation ended max sim", "intention", "asymmetry_count", "n_turns", "asymmetry_intention", "state"])

    # Initialise empty list to store the different arguments in
    list_producer_network = []
    list_interpreter_network = []
    list_intentions = []
    list_n_nodes = []
    list_amount_edges = []
    list_amount_positive_constraints = []
    list_degree_overlap = []
    list_degree_asymmetry = []

    for _ in range(n_runs):

        # First the possible combinations of the amount of edges and positive constraints are used to generate a
        # network
        amount = ["middle", "high"]
        for x in amount:
            amount_edges = x
            amount_positive_constraints = "middle"
            belief_network = CoherenceNetworks(n_nodes, amount_edges, amount_positive_constraints). \
                create_graph()

            # Then the possible combinations of the degree of overlap and asymmetry are used to initialise the
            # network for the producer and interpreter
            degree = [0, 50, 100]
            for i in degree:
                degree_overlap = i
                for n in degree:
                    degree_asymmetry = n
                    belief_network_producer, belief_network_interpreter = initialisation_networks(belief_network,
                                                                                              degree_overlap,
                                                                                              degree_asymmetry)

                    # Randomly generate an intention for the producer
                    possible_intention = []
                    for index in range(n_nodes):
                        if belief_network_producer.nodes[index]['type'] == 'inf':
                            possible_intention.append(index)
                    #print(possible_intention)

                    n_nodes_intention = random.randint(int(0.25 * n_nodes), len(possible_intention))
                    intention = random.sample(possible_intention, k=n_nodes_intention)


                    # Collect arguments
                    list_producer_network.append(belief_network_producer)
                    list_interpreter_network.append(belief_network_interpreter)
                    list_intentions.append(intention)

                    # Collect manipulations to store in dataframe
                    list_n_nodes.append(n_nodes)
                    list_amount_edges.append(amount_edges)
                    list_amount_positive_constraints.append(amount_positive_constraints)
                    list_degree_overlap.append(degree_overlap)
                    list_degree_asymmetry.append(degree_asymmetry)

    # Run a conversation for the specified parameter settings
    pool = multiprocessing.Pool()
    arguments = zip(list_producer_network, list_interpreter_network, list_intentions)
    arg_list = list(arguments)
    result = pool.starmap(conversation, arg_list)
    for index in range(len(list_intentions)):
        result[index]["n_nodes"] = [list_n_nodes[index]] * len(result[index])
        result[index]["amount_edges"] = [list_amount_edges[index]] * len(result[index])
        result[index]["amount_pos_constraint"] = [list_amount_positive_constraints[index]] * len(result[index])
        result[index]["overlap"] = [list_degree_overlap[index]] * len(result[index])
        result[index]["asymmetry"] = [list_degree_asymmetry[index]] * len(result[index])
        result[index]["simulation_number"] = [index] * len(result[index])
        result[index]["ended max sim"] = [result[index].loc[result[index].index[-1], "conversation ended max sim"]] \
                                         * len(result[index])
        results = results.append(result[index])
    pool.close()
    pool.join()

    # Pickle the results
    filename = "results_" + str(n_nodes) + ".p"
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()


def initialisation_networks(belief_network, degree_overlap, degree_asymmetry):
    """
    Initialise the belief networks for the producer and interpreter according to the degree of overlap and asymmetry.
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their constraints as a
    belief network for the producer and interpreter
    :param degree_overlap: int; either 0 (no overlap), 50 (50% overlap), or 100 (complete overlap) of the own beliefs of
    the producer and interpreter
    :param degree_asymmetry: int; either 0 (no asymmetry), 50 (50% asymmetry), or 100 (100% asymmetry) of the
    overlapping own beliefs of the producer and interpreter
    :return: graph; the belief networks for the producer and interpreter
    """

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- Node types ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # First the degree of overlap between the sets of own nodes is set
    n_nodes = belief_network.number_of_nodes()
    if degree_overlap == 100:
        # Choose randomly for the producer which nodes are own beliefs and inferred beliefs with a minimum of 1 own
        # belief node and a maximum of 75% of the nodes
        node_type_producer = random.choices(["own", "inf"], k=n_nodes)
        while node_type_producer.count("own") > (0.75 * n_nodes):
            node_type_producer = random.choices(["own", "inf"], k=n_nodes)
        if "own" not in node_type_producer:
            index = random.randrange(len(node_type_producer))
            node_type_producer[index] = "own"
        # Copy the types for the interpreter as the overlap is 100%
        node_type_interpreter = node_type_producer
    elif degree_overlap == 50:
        # Maximum of 60% can be own beliefs: first randomly choose a percentage under 60 and then randomly choose the
        # corresponding amount of indices to put 'own' in
        # You need at least two own beliefs in order to have 50% overlap
        k = 0
        while k < 2:
            percentage = random.randint(1, 60)
            k = int((percentage / 100) * n_nodes)

        # Here the indices for the producer's own beliefs are randomly chosen
        indices_producer_own = random.sample(list(range(n_nodes)), k=k)

        # Construct the producer's list of node types
        node_type_producer = ["own" if n in indices_producer_own else "inf" for n in range(n_nodes)]

        # Take half of the indices of the producer's own beliefs for the producer
        indices_own_shared = random.sample(indices_producer_own, k=int(0.5 * len(indices_producer_own)))

        # Set these own beliefs in the interpreter's list of node types
        node_type_interpreter = ["own" if n in indices_own_shared else None for n in range(n_nodes)]

        # For the other producer's own beliefs the interpreter needs to have inferred beliefs
        for index in indices_producer_own:
            if index not in indices_own_shared:
                node_type_interpreter[index] = "inf"

        # The own belief set needs to be equal in size for producer and interpreter
        # Count how many own beliefs the producer has
        n_own = node_type_producer.count("own")

        # Count how many own beliefs you still need to set for your interpreter
        n_own_left = n_own - len(indices_own_shared)

        # See which indices are left for the interpreter to set the own beliefs for
        indices_own_left = [i for i in range(len(node_type_interpreter)) if node_type_interpreter[i] is None]

        # Randomly choose which nodes will be the left over own beliefs for the interpreter
        indices_own = random.sample(indices_own_left, k=n_own_left)

        # And set the last own beliefs of the interpreter
        for index in indices_own:
            node_type_interpreter[index] = "own"
    elif degree_overlap == 0:
        # A maximum of half of the beliefs can be own in order not to have any overlap and you need at least one own
        # belief
        k = 0
        while k < 1:
            percentage = random.randint(1, 50)
            k = int((percentage / 100) * n_nodes)

        # Here the indices for the producer's own beliefs are randomly chosen
        indices_producer_own = random.sample(list(range(n_nodes)), k=k)

        # Construct the producer's list of node types
        node_type_producer = ["own" if n in indices_producer_own else "inf" for n in range(n_nodes)]

        # Get the indices for the producer's inferred node types
        indices_producer_inf = [i for i in range(len(node_type_producer)) if node_type_producer[i] == "inf"]

        # Get the number of own beliefs for the interpreter according to the producer
        n_own_producer = len(node_type_producer) - len(indices_producer_inf)

        # Randomly choose which nodes will be the own beliefs for the interpreter (i.e., where the producer has inferred
        # nodes)
        indices_own_interpreter = random.sample(indices_producer_inf, k=n_own_producer)

        # Set the node types for the interpreter
        node_type_interpreter = ["own" if i in indices_own_interpreter else "inf" for i in range(len(node_type_producer))]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- Truth values ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Independently generate the truth values for the nodes for the producer and interpreter
    truth_value_producer = random.choices([True, False], k=n_nodes)
    truth_value_interpreter = random.choices([True, False], k=n_nodes)

    # Create a list which contains all the indices of the overlapping own beliefs
    indices_own_shared = []
    for index in range(len(node_type_producer)):
        if node_type_producer[index] == node_type_interpreter[index] and node_type_producer[index] == "own":
            indices_own_shared.append(index)

    # Make the truth values of the overlapping own beliefs the same
    for index in indices_own_shared:
        if truth_value_producer[index] != truth_value_interpreter[index]:
            truth_value_interpreter[index] = truth_value_producer[index]
    # If the degree of asymmetry is 100, change all the interpreter's overlapping own truth values to the opposite
    # of the producer's
    if degree_asymmetry == 100:
        for index in indices_own_shared:
            truth_value_interpreter[index] = not truth_value_producer[index]
    # If the degree of asymmetry is 50, change half of the interpreter's overlapping own truth values to the opposite
    # of the producer's
    if degree_asymmetry == 50:
        k = int(len(indices_own_shared) / 2)
        flip_indices = random.sample(indices_own_shared, k=k)
        for index in flip_indices:
            truth_value_interpreter[index] = not truth_value_producer[index]

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Initialisation ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Initialisation of producer and interpreter belief_network
    belief_network_producer = initialisation_network(belief_network, node_type_producer, truth_value_producer,
                                                    "producer")
    belief_network_interpreter = initialisation_network(belief_network.copy(), node_type_interpreter,
                                                     truth_value_interpreter,
                                                     "interpreter")
    return belief_network_producer, belief_network_interpreter


def initialisation_network(belief_network, node_type, node_truth_value, agent):
    """
    Initialisation of belief belief_network with the starting node types and truth values for the nodes.
    :param agent: string; the agent type for the network
    :param belief_network: graph; the graph containing the relevant nodes connected by edges with their
    constraints as a belief network for the interpreter and producer
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

    # If the agent is a interpreter a node attribute should be initialised whether repair was already asked over that node
    if agent == "interpreter":
        # Initialise all the nodes of the interpreter with repair set to false as repair has not been used yet
        nx.set_node_attributes(belief_network, False, "repair")

    return belief_network
