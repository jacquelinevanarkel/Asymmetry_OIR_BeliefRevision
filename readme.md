## How do people resolve misunderstanding in conversation through repair? Introducing a framework built on belief revision and coherence

The implementation of the **(i) model**, **(ii) the simulations** and **(iii) the results** described in this master's thesis can be found here. In order to access the data resulting from the simulations run, I redirect you to [this OSF page of this project](https://osf.io/xah9u/?view_only=3a4737cb6be3461f853501ed11c0b704).

(i) The implementation of **the model** consists out of three files:
1. **coherence_networks.py**: the implementation for generating coherence networks as belief networks for the agents.
2. **interpreter.py**: the implementation of the interpreter's model.
3. **producer.py**: the implementation of the producer's model.

(ii) In order to run **the simulations** of conversations between two agents, the file 'simulation.py' can be used. Here, the agents' networks get initialised, a conversation is simulated, and multiple conversations can be run. In order to start the simulations, the function 'simulation(n_nodes, n_runs)' should be called. n_nodes stands for the number of nodes in the agents' networks and n_runs for how many conversations should be ran. The resulting dataframe is pickled and stored in the local directory of the simulation.py file.

(iii) In order to study **the results**, some plotting code is provided to recreate the plots used in this thesis. The file 'plotting.py' contains the necessary code to recreate these plots. Remember to change the path to read in the data to the folder where you have stored the data.
(Some old plots used for exploration have been discarded, some old ones still exist in 'plotting_old.py', however, this file is outdated. It can serve as inspiration or help, but it might need some fixing first.)
