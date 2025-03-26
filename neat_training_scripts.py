import jax, jax.numpy as jnp
from tensorneat.problem import BaseProblem
from tensorneat.common import State
from collections import deque
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, RecurrentGenome
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import pandas as pd

# to remove pandas error
pd.options.mode.chained_assignment = None




def read_dataset(datasets : str):
    data = pd.read_csv(datasets)
    
    data_input = data[["x", "y","x_to","y_to", "dt"]]
    # data_input['xlag_2'] = data["x"].shift(2)
    # data_input['ylag_2'] = data["y"].shift(2)
    # # data_input['dxlag'] = data["dx"].shift(1)
    # data_input['dylag'] = data["dy"].shift(1)
    # data_input["dxlaglag"] = data["dx"].shift(2)
    # data_input["dylaglag"] = data["dy"].shift(2)
    # data_input["dxlaglaglag"] = data["dx"].shift(3)
    # data_input["dylaglaglag"] = data["dy"].shift(3)

    data_input.fillna(0, inplace=True)

    # env_input = jnp.array(data_input.to_numpy())
    # displacement = jnp.array(data[["dx", "dy"]].to_numpy())
    env_input = data_input.to_numpy()

    displacement = data[['dx', 'dy']]

    return env_input, displacement

# Define the custom Problem
class cloneProblem(BaseProblem):

    jitable = True
    alpha = 0.5 # best 0.5
    loss = "aar"
        # actions
    def get_data(self, env_input, displacement):
        self.input = env_input
        self.displacement = displacement


    def evaluate(self, state, randkey, act_func, params):
        # Use ``act_func(state, params, inputs)`` to do network forward
        # the loss is define as the Action agreement ratio, where if the action is the same, the loss is 1.0
        # do batch forward for all inputs (using jax.vamp)
        disp_predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.input
        )  # should be shape (1000, 1)

        if self.loss == "aar":
            # print("here")
            dist = jnp.linalg.norm(self.displacement - disp_predict)
            loss = 1/(self.alpha*dist + 1.0) # continous action agreement
            # loss = jnp.exp(-dist)
            loss = jnp.mean(loss)
        else:
            # mse loss
            loss = -jnp.mean(jnp.square(self.displacement - disp_predict))

        # return negative loss as fitness 
        # TensorNEAT maximizes fitness, equivalent to minimizes loss
        return loss

    @property
    def input_shape(self):
        # the input shape that the act_func expects
        return (5, )
    
    @property
    def output_shape(self):
        # the output shape that the act_func returns
        return (2, )
    
    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # shocase the performance of one individual
        disp_predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.input
        )  # should be shape (1000, 1)
        
        if self.loss == "aar":
            # # my vision of action agreement ratio
            dist = jnp.linalg.norm(disp_predict - self.displacement, axis=1)
            loss = 1/(dist + 1.0) # continous action agreement
            loss = jnp.mean(loss)
        else:
            # mse loss
            loss = jnp.mean(jnp.square(disp_predict-self.displacement))

        msg = ""
        for i in range(self.input.shape[0]):
            msg += f"input: {self.input[i]}, target: {self.displacement[i]}, predict: {disp_predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)



def validate_genome(pipeline, state, best , scaler, plot=False):
    print("\n\n------------------------   Validation ! ------------------------ ")
    pipeline.show(state, best)
    

    
    if plot:
        # get genome
        genome = pipeline.algorithm.genome
        state = State(randkey=jax.random.key(0))

        genome.setup(state)

        transformed = genome.transform(state, *best)
        act_fun = jax.jit(genome.forward)

        n = problem.input.shape[0]
        # n = 5
        xc = problem.input[0]
        positions = [jax.device_get(xc[0:2])]
        for i in range(n):
            
            jax_d = act_fun(state, transformed, xc)


            unscale_position = scaler.inverse_transform(jax.device_get(xc).reshape(1, xc.shape[0]))
            new_unscale_position = unscale_position[0, 0:2] + jax.device_get(jax_d)
            scale_position = scaler.transform(jnp.hstack((new_unscale_position ,np.array([0,0,0]))).reshape(1, xc.shape[0]))
            
            positions.append(scale_position[0][0:2])
            xc = jax.device_get(problem.input[i+1])
            xc = jnp.array([scale_position[0][0], scale_position[0][1], xc[2], xc[3], xc[4]])

    return positions

if __name__ == "__main__":

    name = "C0_P0"
    # log_dir = "clone/" +datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = "clone_log/" + name

    config_file = "neat_configfile.yaml"

    with open(config_file, 'r') as file:
            # Load the YAML content
            config = yaml.safe_load(file)

    # Access the configuration values
    neat_config = config['NEAT']
    pipeline_config = config['pipeline']

    # Print the configuration values
    print("NEAT Configuration:")
    print(f"Population Size: {neat_config['pop_size']}")
    print(f"Species Size: {neat_config['pop_size']}")
    print(f"Survival Threshold: {neat_config['survival_threshold']}")

    print("\nPipeline Configuration:")
    print(f"Generation Max: {pipeline_config['gen_max']}")
    print(f"Fitness Target: {pipeline_config['fitness_target']}")
    print(f"Seed: {pipeline_config['seed']}")


    problem = cloneProblem()

    env_input, displacement = read_dataset("datasets/P0_C0.csv")

    # scale datas
    scaler = StandardScaler()
    env_input = scaler.fit_transform(env_input)

    x = jnp.array(env_input)
    y = jnp.array(displacement.to_numpy())
    problem.get_data(x, y)




    algorithm = NEAT(
        pop_size=neat_config['pop_size'], #best 500 
        species_size=neat_config['pop_size'],
        survival_threshold=neat_config['survival_threshold'],
        genome=RecurrentGenome (
            num_inputs=5,
            num_outputs=2,
            max_nodes=50,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=[ACT.identity],
                aggregation_options=[AGG.sum],
            ),
            output_transform=ACT.identity,
        ),
    )




    # problem.evaluate(state, randkey, act_func, params)
    # Construct the pipeline and run
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        generation_limit=pipeline_config['gen_max'],
        fitness_target=pipeline_config['fitness_target'],
        seed=pipeline_config['seed'],
        is_save=True,
        save_dir=log_dir,
        show_problem_details=True
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    net = pipeline.algorithm.genome.network_dict(state, *best)
    # pipeline.algorithm.genome.visualize(net, save_path=log_dir + '/net.png')
    positions = validate_genome(pipeline, state, best, scaler, True)

    positions = np.array(positions)

    plt.plot(positions[:, 0], positions[:, 1], '.', label="agent movement")
    plt.plot(env_input[:, 0], env_input[:, 1], label="ground truth")
    plt.show()