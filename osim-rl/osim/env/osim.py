import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
import pandas as pd
import plugins.my_manager_factory
import matplotlib.pyplot as plt


## OpenSim interface
# The amin purpose of this class is to provide wrap all 
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class

class OsimModel(object):
    # Initialize simulation
    stepsize = 0.01


    start_point = 0
    model = None
    state = None
    state0 = None
    starting_speed = None

    k_path = None # path to CMC tracking states file
    k_paths_dict = {}
    states = None
    states_trajectories_dict = {}
    walk = True
    init_traj_ix = None

    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0
    
    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    maxforces = []
    curforces = []

    # Takes a dict where the keys are the speeds of the corresponding paths pointing to motion clips

    def __init__(self, model_path, k_paths_dict, visualize, integrator_accuracy = 1e-3): # Reduced accuracy for greater speed
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        self.k_path = k_paths_dict[list(k_paths_dict.keys())[0]]
        self.states = pd.read_csv(self.k_path, index_col=False).drop('Unnamed: 0', axis=1)
        for speed in list(k_paths_dict.keys()):
            self.states_trajectories_dict[speed] = pd.read_csv(k_paths_dict[speed], index_col=False).drop('Unnamed: 0', axis=1)

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        self.actuatorSet = self.model.getActuators()

        if self.verbose:
            self.list_elements()

        for j in range(self.actuatorSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.actuatorSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.curforces.append(1.0)

            try:
                self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())

            except:
                self.maxforces.append(np.inf)


        self.noutput = self.actuatorSet.getSize()     
            
        self.model.addController(self.brain)
        self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i,self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i,self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i,self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i,self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i,self.markerSet.get(i).getName())
        print("")

    def actuate(self, action):

        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        # TODO: Check if actions within [0,1]
        self.last_action = action
            
        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):


            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    """
    Directly modifies activations in the current state.
    """
    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """
    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]


    """
    This function defines what we can later construct the state from in each environmnet
    """
    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html 
        
        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    # def reset_manager(self):
    #     self.manager = opensim.Manager(self.model)
    #     self.manager.setIntegratorAccuracy(self.integrator_accuracy)
    #     self.manager.initialize(self.state)

    # Modified integrator - roughly 2x speedup
    def reset_manager(self):
        self.manager = opensim.Manager()
        type_ = plugins.my_manager_factory.integrator.SemiExplicitEuler2
        plugins.my_manager_factory.my_manager_factory.create(self.model, self.manager,type_,self.integrator_accuracy)
        self.manager.initialize(self.state)


    def multi_clip_reset(self, test):

        """
        If the initialisation speed is not 1.25, never init into the standing position.
        If it is 1.25, mostly go into standing, sometimes RSI
        Do not need to learn to start at other speeds
        """

        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.istep = self.start_point =  0
        self.init_traj_ix = -1


        if test == False:
            if self.starting_speed != 1.25 or np.random.rand()>0.8: 


                init_data = self.states_trajectories_dict[self.starting_speed]
                self.istep = self.start_point =  np.random.randint(20,150) 
                init_states = init_data.iloc[self.istep,1:].values
                vec = opensim.Vector(init_states)
                self.model.setStateVariableValues(self.state, vec)
        
        self.state.setTime(self.istep*self.stepsize)
        self.reset_manager()


    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        try:
            self.state = self.manager.integrate(self.stepsize * self.istep)
        except Exception as e:
            print (e)




class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 900

## OpenAI interface
# The amin purpose of this class is to provide wrap all 
# the functions of OpenAI gym. It is still an abstract
# class but closer to OpenSim. The actual classes of
# environments inherit from this one and:
# - select the model file
# - define the rewards and stopping conditions
# - define an obsernvation as a function of state
class OsimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False
    reward_speed = None
    visualize = False
    spec = None
    time_limit = 1e10

    prev_state_desc = None

    model_path = None 
    k_path = None
    k_path2 = None
    k_paths = []

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : None
    }

    def reward(self):
        raise NotImplementedError

    def is_done(self):
        return False

    def __init__(self, visualize = True, integrator_accuracy = 1e-3):
        self.osim_model = OsimModel(self.model_path, self.k_paths_dict, visualize, integrator_accuracy = integrator_accuracy)

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

        if not self.action_space:
            self.action_space = ( [0.0] * self.osim_model.get_action_space_size(), [1.0] * self.osim_model.get_action_space_size() )
        if not self.observation_space:
            self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)




    def get_state_desc(self):
        return self.osim_model.get_state_desc()

    def get_prev_state_desc(self):
        return self.prev_state_desc

    def get_observation(self):
        # This one will normally be overwrtitten by the environments
        # In particular, for the gym we want a vector and not a dictionary
        return self.osim_model.get_state_desc()

    def get_observation_space_size(self):
        return 0

    def get_action_space_size(self):
        return self.osim_model.get_action_space_size()

    def reset(self, project = True):
        self.osim_model.reset()
        
        if not project:
            return self.get_state_desc()
        return self.get_observation()

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        return [ obs, self.reward(self.osim_model.istep), self.is_done() or (self.osim_model.istep >= (self.spec.timestep_limit+self.osim_model.start_point)), {} ]

    def render(self, mode='human', close=False):
        return



class ProstheticsEnv_R2_multiclip(OsimEnv):
    """
    New env allowing training with multiple clips.
    Each episode (triggered by env specific reset function) pick one of the clips to initialise into (by RSI).
    Target speed is predetermined by generate targets funtion, and this feeds into both the state input and the reward function.
    At each timestep, take max reward over all clips to encourage learning from all of them or select the clip closest in speed
    """
    

    model_path = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_pros_20180507.osim')     
    k_path = os.path.join(os.path.dirname(__file__), '../data/075.osim')     
    k_path2 = os.path.join(os.path.dirname(__file__), '../data/125.osim')     
    k_path3 = os.path.join(os.path.dirname(__file__), '../data/175.osim')     


    k_paths = [k_path, k_path2, k_path3]
    k_paths_dict = {}
    k_paths_dict[0.75] = k_path
    k_paths_dict[1.25] = k_path2
    k_paths_dict[1.75] = k_path3


    time_limit = 890 
    difficulty=1

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit
        self.spec.timestep_limit = time_limit


    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 0:
            self.time_limit = 300
        if difficulty == 1:
            self.time_limit = 1000
        self.spec.timestep_limit = self.time_limit   

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.7 

    def get_observation(self):
        state_desc = self.get_state_desc()

        res = []


        res += [state_desc["target_vel"][0]- state_desc["body_vel"]['pelvis'][0]]
        res += [state_desc["target_vel"][2]- state_desc["body_vel"]['pelvis'][2]]


        res += [state_desc["target_vel"][0]]
        res += [state_desc["target_vel"][2]]

        
        pelvis_x_pos = state_desc["body_pos"]["pelvis"][0]
        pelvis_y_pos = state_desc["body_pos"]["pelvis"][1]
        pelvis_z_pos = state_desc["body_pos"]["pelvis"][2]

        for body_part in ["pelvis"]:
            res += state_desc["body_pos_rot"][body_part][:3] 
            res += state_desc["body_vel_rot"][body_part][:3]
            res += state_desc["body_acc_rot"][body_part][:3]
            res += state_desc["body_acc"][body_part][0:3]


            res += [state_desc["body_vel"][body_part][0]]
            res += [pelvis_y_pos]
            res += [state_desc["body_vel"][body_part][1]]
            res += [state_desc["body_vel"][body_part][2]]

        for body_part in ["head","torso", "pros_tibia_r","pros_foot_r","toes_l","talus_l"]:
            res += state_desc["body_pos_rot"][body_part][:3] 
            res += state_desc["body_vel_rot"][body_part][:3]
            res += state_desc["body_acc_rot"][body_part][:3]
            res += state_desc["body_acc"][body_part][:3]
            res += [state_desc["body_pos"][body_part][0] - pelvis_x_pos]
            res += [state_desc["body_vel"][body_part][0]]
            res += [state_desc["body_pos"][body_part][1] - pelvis_y_pos]
            res += [state_desc["body_vel"][body_part][1]]
            res += [state_desc["body_pos"][body_part][2] - pelvis_z_pos]
            res += [state_desc["body_vel"][body_part][2]]


        for joint in ["hip_r","knee_r","hip_l","knee_l","ankle_l"]: 
            res += state_desc["joint_pos"][joint][:2]
            res += state_desc["joint_vel"][joint][:2]
            res += state_desc["joint_acc"][joint][:2] 

        for muscle in state_desc["muscles"].keys():
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["forces"]["ankleSpring"]

        for foot in ['pros_foot_r_0','foot_l']:
            res += state_desc['forces'][foot][:6]

        cm_pos_x = [state_desc["misc"]["mass_center_pos"][0] - pelvis_x_pos]
        cm_pos_y = [state_desc["misc"]["mass_center_pos"][1] - pelvis_y_pos]
        cm_pos_z = [state_desc["misc"]["mass_center_pos"][2] - pelvis_z_pos]
        res = res + cm_pos_x + cm_pos_y + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        self.state_buffer.append(res)

        return res

    def generate_new_targets(self, test, poisson_lambda = 300):
        nsteps = self.time_limit + 1 + 150 
        rg = np.array(range(nsteps))
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        velocities = sorted(list(self.k_paths_dict.keys()))

        if test == True or np.random.rand() > 0.5
            velocity [0] = 1.25

        else:
            trimmed_velocities = velocities.copy()
            trimmed_velocities.remove(1.25)
            velocity[0] = trimmed_velocities[np.random.choice(len(trimmed_velocities))]

        self.osim_model.starting_speed = velocity[0]
        heading[0] = 0

        change = np.cumsum(np.random.poisson(poisson_lambda, 10))

        for i in range(1,nsteps):
            velocity[i] = velocity[i-1]
            heading[i] = heading[i-1]

            if i in change:

                velocity[i] += random.choice([-1,1]) * random.uniform(-0.5,0.5)
                heading[i] += random.choice([-1,1]) * random.uniform(-math.pi/8,math.pi/8)

        

        trajectory_polar = np.vstack((velocity,heading)).transpose()
        self.targets = np.apply_along_axis(rect, 1, trajectory_polar)

        
    def get_state_desc(self):
        d = super(ProstheticsEnv_R2_multiclip, self).get_state_desc()
        if self.difficulty > 0:
            d["target_vel"] = self.targets[self.osim_model.istep,:].tolist()
        return d

    def get_observation_space_size(self):
        return 227 

    def compute_reward(self, t, kin_df):

        self.kins = kin_df
        state_desc = self.get_state_desc()

        pelvis_loss =  (state_desc["body_pos"]["pelvis"][0] - self.kins['pelvis_tx'][t])**2 + (state_desc["body_pos"]["pelvis"][1] - self.kins[
            'pelvis_ty'][t])**2 + (state_desc["body_pos"]["pelvis"][2] - self.kins['pelvis_tz'][t])**2
        pelvis_rot_loss = (state_desc["joint_pos"]["ground_pelvis"][1] - self.kins['pelvis_list'][t])**2 + (state_desc["joint_pos"]["ground_pelvis"][2] - self.kins[
            'pelvis_rotation'][t])**2 + (state_desc["joint_pos"]["ground_pelvis"][0] - self.kins['pelvis_tilt'][t])**2

        ankle_loss = (state_desc['joint_pos']['ankle_l'] - self.kins['ankle_angle_l'][t])**2
        knee_loss = (state_desc['joint_pos']['knee_l'] -self.kins['knee_angle_l'][t])**2 + (state_desc['joint_pos']['knee_r'] - self.kins['knee_angle_r'][t])**2
        hip_loss = (state_desc['joint_pos']['hip_l'][0] - self.kins['hip_flexion_l'][t])**2 + (state_desc['joint_pos']['hip_r'][0] - self.kins[
            'hip_flexion_r'][t])**2 + (state_desc['joint_pos']['hip_l'][1] - self.kins['hip_adduction_l'][t])**2 + (state_desc['joint_pos']['hip_r'][1] - self.kins['hip_adduction_r'][t])**2
            
        pelvis_x_pos = state_desc["body_pos"]["pelvis"][0]
        pelvis_y_pos = state_desc["body_pos"]["pelvis"][1]
        pelvis_z_pos = state_desc["body_pos"]["pelvis"][2]

        ankle_loss = (state_desc['joint_pos']['ankle_l'] - self.kins['ankle_angle_l'][t])**2
        total_position_loss = ankle_loss  + knee_loss + hip_loss 
        pos_reward = np.exp(-2* total_position_loss) 

        # velocity losses
        pelvis_rot_loss_v = (state_desc["joint_vel"]["ground_pelvis"][0] - self.kins['pelvis_tilt_speed'][t])**2 + (state_desc["joint_vel"]['ground_pelvis'][2] - self.kins[
            'pelvis_rotation_speed'][t])**2 + (state_desc["joint_vel"]["ground_pelvis"][1] - self.kins['pelvis_list_speed'][t])**2
        pelvis_loss_v = (state_desc["body_vel"]['pelvis'][0] - self.kins['pelvis_tx_speed'][t])**2 + (state_desc["body_vel"]['pelvis'][1] - self.kins[
            'pelvis_ty_speed'][t])**2 +(state_desc["body_vel"]['pelvis'][2] - self.kins['pelvis_tz_speed'][t])**2 
        ankle_loss_v = (state_desc['joint_vel']['ankle_l'] - self.kins['ankle_angle_l_speed'][t])**2 
        knee_loss_v = (state_desc['joint_vel']['knee_l'] -self.kins['knee_angle_l_speed'][t])**2 + (state_desc['joint_vel']['knee_r'] - self.kins['knee_angle_r_speed'][t])**2
        hip_loss_v = (state_desc['joint_vel']['hip_l'][0] - self.kins['hip_flexion_l_speed'][t])**2 + (state_desc['joint_vel']['hip_r'][0] - self.kins[
            'hip_flexion_r_speed'][t])**2 + (state_desc['joint_vel']['hip_l'][1] - self.kins['hip_adduction_l_speed'][t])**2 + (state_desc['joint_vel']['hip_r'][1] - self.kins['hip_adduction_r_speed'][t])**2

        total_velocity_loss = ankle_loss_v + knee_loss_v + hip_loss_v 
        velocity_reward = np.exp(-0.1*total_velocity_loss) 

        im_rew =  0.75*pos_reward + 0.25*velocity_reward
        return im_rew[0]


    def reward(self,t):
 

        state_desc = self.get_state_desc()

        target_x_speed = self.targets[t,0]
        target_z_speed = self.targets[t,2]

        x_penalty = (state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0])**2
        z_penalty = (state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2])**2

        goal_rew = np.exp(-8*(x_penalty + z_penalty))

        penalty = 0
        penalty += np.sum(np.array(self.osim_model.get_activations())**2) * 0.001

        penalty += x_penalty
        penalty += z_penalty

        target_velocity = np.sqrt(target_x_speed**2 + target_z_speed**2)
        np_speeds = np.array(list(self.k_paths_dict.keys()))
        closest_clip_ix = (np.abs(np_speeds - target_velocity)).argmin()
        closest_clip_speed = np_speeds[closest_clip_ix]
        im_rew = self.compute_reward(t, self.osim_model.states_trajectories_dict[closest_clip_speed])

        if t<=50:
            return (0.1*im_rew + 0.9* goal_rew, 10-penalty) 

        return (0.7*im_rew + 0.3* goal_rew, 10-penalty) 


    def reset(self, test, project = True):

        self.generate_new_targets(test)
        self.osim_model.multi_clip_reset(test)


        if not project:
            return self.get_state_desc()
        return self.get_observation()

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        return [ obs, self.reward(self.osim_model.istep)[0], self.reward(self.osim_model.istep)[1], self.is_done() or (self.osim_model.istep >= (self.spec.timestep_limit+self.osim_model.start_point)), {} ]

