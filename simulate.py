import os
import importlib

########### Set PARAMS here ############
TIME_OF_DAY = 17.0 # military time
CLOUD_COVER = 0  #0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
RUNWAY = 'KMWH'
START_POS = [-6, 110.0, -17.0] # [CTE, DTP, HE]
USE_FALLBACK = True

sim_helper_dir = "sim_helper"
sim_helper_filename = 'sim_helper.simulator_utils_{}.py'.format(RUNWAY)
sim_helper_file_path = os.path.join(sim_helper_dir, sim_helper_filename)

if os.path.exists(sim_helper_file_path):
    spec = importlib.util.spec_from_file_location(sim_helper_filename, sim_helper_file_path)
    sim_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_utils)

foldername = 'simulation'
save_traj = 0
ideal = 0 # 1 for ideal controller 0 for NN-controller
freq = 1
dt = 0.05
tMax = 10.0
trajectory_plot = 1

MODEL_WEIGHTS = 'weights/taxinet_classifier_weights.pth'

##########################################
num_steps = int(tMax/dt)
start_params = {'foldername': foldername,
                'save_traj': save_traj,
                'ideal': ideal,
                'freq': freq,
                'TIME_OF_DAY': TIME_OF_DAY,
                'CLOUD_COVER': CLOUD_COVER,
                'start': START_POS,
                'dt': dt,
                'tMax': tMax,
                'num_steps': num_steps,
                'control_every': 1,
                'sim_type': 'dubins'
                }


client, network, classifier = sim_utils.get_simulator_and_model(start_params, MODEL_WEIGHTS)

# Start the simulation
xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, predicted_images, predicted_downsampled_images, xs_record, he_record, ys_record, velocities = rollout(client, network, 
                                                                                                                num_steps = num_steps, 
                                                                                                                dt = dt, 
                                                                                                                ideal=ideal,
                                                                                                                classifier = classifier,
                                                                                                                use_fallback = USE_FALLBACK,
                                                                                                                )

if save_traj:
    sim_utils.save_metrics(xs, ys, thetas, pred_xs, pred_thetas, control, orig_images, downsampled_images, result, episode_steps, start_params, predicted_images, predicted_downsampled_images)

if trajectory_plot:
    sim_utils.plot_trajectory(xs_record, ys_record, USE_FALLBACK)