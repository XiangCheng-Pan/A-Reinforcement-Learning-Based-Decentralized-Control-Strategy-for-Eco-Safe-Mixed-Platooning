import torch
import numpy as np
import scipy.io
from torch.utils.tensorboard import SummaryWriter

import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from Comm_PPO import PPO_continuous

import optparse
import traci
import matplotlib.pyplot as plt
import datetime

episode_num = 0
episode_max = 500
step_max = 250
BATCH = 16

import sys
import os
import math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable ")

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def state_calc(vehicle, position, speed, acc):
    state = [0] * 4
    state[0] = (speed[vehicle] - speed[vehicle + 1]) / 20
    state[1] = (speed[0] - speed[vehicle + 1]) / 20
    state[2] = speed[vehicle + 1] / 20
    state[3] = (position[vehicle] - position[vehicle + 1]) / 40
    return state

def reward_calc(vehicle, position, speed, acc, chongtu, chaoguo,acc_last = [], Platoonsize_Max = 16):

    headway = (position[vehicle] - position[vehicle + 1])
    energy = 0
    discount = 1
    for i in range(Platoonsize_Max - vehicle):
        a = acc[vehicle + i + 1]
        energy += a ** 2 / 9 * discount
        discount *= 0.4

    if headway > 50:
        energy += headway / 50

    if chongtu == 1 or chaoguo == 1:
        energy += 1
    energy *= 100
    return -energy



def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, seed):

    speedmode = 6
    madr = 1.4
    sumoBinary = checkBinary('sumo')
    # sumoBinary = checkBinary('sumo-gui')
    speed_init = 20

    leading = []
    for i in range(0, 40):
        leading.append(0)
    for i in range(40, 65):
        leading.append(-1)
    for i in range(65, 150):
        leading.append(1)
    for i in range(150, 300):
        leading.append(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    episode_reward = [0]

    args.state_dim = 4
    args.action_dim = 1
    args.max_action = 3
    args.max_episode_steps = 200

    replay_buffer = ReplayBuffer(args)

    ppo = PPO_continuous(args)
    path1 = './models/Compare-comm-ppo-Actor.pth'
    path2 = './models/Compare-comm-ppo-Critic.pth'
    ppo.load_model(path1,path2)

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)


    Platoonsize = 16
    args.batch_size = Platoonsize * 16
    args.mini_batch_size = Platoonsize
    replay_buffer = ReplayBuffer(args)
    ppo.batch_size = Platoonsize * 16
    ppo.mini_batch_size = Platoonsize

    Position = [0] * (Platoonsize + 1)
    Speed = [0] * (Platoonsize + 1)
    Acc = [0] * (Platoonsize + 1)

    position = [[]for i in range(Platoonsize + 1)]
    speed = [[]for i in range(Platoonsize + 1)]
    acclererate = [[]for i in range(Platoonsize + 1)]
    time = [[]for i in range(Platoonsize + 1)]
    position_error =  [[]for i in range(Platoonsize + 1)]
    speed_error =  [[]for i in range(Platoonsize + 1)]
    acc_error =  [[]for i in range(Platoonsize + 1)]

    position_plot = []
    speed_plot = []
    acc_plot = []
    time_plot = []
    position_error_plot = []

    done = False

    consumption = 0
    distance = 0

    reward_ave = 0

    # traci.start([sumoBinary, "-c", "car_str.sumocfg"])
    traci.start([sumoBinary, "-c", "car_str_16.sumocfg"])
    state = [[] for i in range(Platoonsize + 1)]
    reward = [[] for i in range(Platoonsize + 1)]
    action = [[] for i in range(Platoonsize + 1)]
    action_logprob = [[] for i in range(Platoonsize + 1)]
    state_next = [[] for i in range(Platoonsize + 1)]
    buffer_reward = [[] for i in range(Platoonsize + 1)]
    traci.simulationStep()
    controlled_vehicles = ['a', 'b.0', 'b.1', 'b.5', 'b.6', 'b.7', 'c.1', 'c.3', 'c.5', 'c.6']
    exist_list = traci.vehicle.getIDList()
    for car in exist_list:
        ind = exist_list.index(car)
        if ind <= Platoonsize:
            if car in controlled_vehicles:
                traci.vehicle.setMinGap(car, 0)
                traci.vehicle.setSpeedMode(car, speedmode)
            else:
                traci.vehicle.setSpeedMode(car, 0)
            # traci.vehicle.setMinGap(car,0)
            # traci.vehicle.setSpeedMode(car, speedmode)
            Position[ind] = traci.vehicle.getPosition(car)[0]
            Speed[ind] = round(traci.vehicle.getSpeed(car), 1)
            Acc[ind] = traci.vehicle.getAcceleration(car)
    accelerate_accepted = [3] * (Platoonsize + 1)

    for step in range(step_max):

        for i in range(Platoonsize + 1):
            position_plot.append(Position[i] / 1000)
            speed_plot.append(Speed[i])
            time_plot.append(step)
            acc_plot.append(Acc[i])

            position[i].append(Position[i])
            speed[i].append(Speed[i])
            acclererate[i].append(Acc[i])
            if i == 0:
                position_error_plot.append(0)
            if i >= 1:
                position_error[i].append(Position[i-1]-Position[i])
                position_error_plot.append(Position[i - 1] - Position[i])
                speed_error[i].append(Speed[i-1]-Speed[i])
                acc_error[i].append(Acc[i]-Acc[0])


        for i in range(Platoonsize):
            if Speed[i] - 3 < Speed[i + 1]:
                gap = Position[i] - Position[i + 1] - 5 - Speed[i + 1] + max(Speed[i] - 3, 0)
                if gap < 0:
                    amax = -3
                else:
                    amax = min(gap / 3, math.sqrt(madr * gap)) + Speed[i] - Speed[i + 1] - 3
                    amax = np.clip(amax, -3, 3)
            else:
                amax = 3

            accelerate_accepted[i + 1] = amax

        acc_matrix = [0] * (Platoonsize + 1)


        for i in range(1, Platoonsize + 1):
            state[i].append(state_calc(i - 1, Position, Speed, Acc))

        if args.use_state_norm:
            for i in range(1, Platoonsize + 1):
                state[i][-1] = state_norm(state[i][-1])

        if args.use_reward_scaling:
            reward_scaling.reset()

        for i in range(1, Platoonsize + 1):
            # act, act_log_prob = ppo.choose_action(np.array(state[i][-1]))
            act = ppo.evaluate(np.array(state[i][-1]))
            if args.policy_dist == "Beta":
                act = 3 * (act - 0.5) * args.max_action  # [0,1]->[-max,max][-3,3]

            action[i].append(act)
            # action_logprob[i].append(act_log_prob)

        for i in range(1, Platoonsize + 1):
            acc_matrix[i] = action[i][-1][0]

        pos_beyond = [0] * (Platoonsize + 1)
        pos_conflict = [0] * (Platoonsize + 1)
        # 作者在原文这里添加了领航车的速度处理
        speed_next = np.clip(Speed[0] + leading[step], 0, speed_init)
        traci.vehicle.setSpeed(exist_list[0], speed_next)

        for i in range(1, Platoonsize + 1):
            car = exist_list[i]
            if car in controlled_vehicles:
                accc = min(acc_matrix[i], accelerate_accepted[i])
                if acc_matrix[i] > accelerate_accepted[i] + 0.5:
                    pos_beyond[i] = 1
                speed_next = np.clip(Speed[i] + accc, 0, 35)
                traci.vehicle.setSpeed(car, speed_next)
            else:
                pass

        Acc_last = Acc
        Position = [0] * (Platoonsize + 1)
        Speed = [0] * (Platoonsize + 1)
        Acc = [0] * (Platoonsize + 1)

        traci.simulationStep()
        exist_list = traci.vehicle.getIDList()
        if 'a' not in exist_list:
            break

        for car in exist_list:
            ind = exist_list.index(car)
            # print(car)
            if ind <= Platoonsize:
                Position[ind] = traci.vehicle.getPosition(car)[0]
                Speed[ind] = round(traci.vehicle.getSpeed(car), 1)
                Acc[ind] = traci.vehicle.getAcceleration(car)

        for i in range(1, Platoonsize + 1):
            # 这里判断是否发生碰撞，碰撞的话车会消失，所以一般要结束进程了，或许前期可以通过算法挽回以下，让他跑更多的时间步
            if i > 0 and (Position[i] > Position[i - 1] - 5 or Position[i] < -10000):
                pos_conflict[i] = 1
        if sum(pos_conflict) > 0:
            print('车辆发生碰撞')
            print(step, pos_conflict)
            break

    traci.close()

    plt.figure(1)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(time_plot, position_plot, c=speed_plot, s=10, alpha=0.3)
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Location (km)')
    plt.grid(True)
    plt.show()

    data1 = np.column_stack((time_plot,position_plot,speed_plot))
    np.savetxt('.\\output\\position-100.csv', data1, delimiter=',', header='Time,Location,Speed', comments='')

    time_plot_array = np.array(time_plot).reshape(-1, Platoonsize + 1)
    position_plot_array = np.array(position_plot).reshape(-1, Platoonsize + 1)
    speed_plot_array = np.array(speed_plot).reshape(-1, Platoonsize + 1)
    acc_plot_array = np.array(acc_plot).reshape(-1, Platoonsize + 1)
    position_error_plot_array = np.array(position_error_plot).reshape(-1, Platoonsize + 1)

    data2 = np.column_stack((time_plot_array, position_plot_array, speed_plot_array, acc_plot_array, position_error_plot_array))

    header = ','.join([f'Time_{i}' for i in range(Platoonsize + 1)])
    header += ',' + ','.join([f'Position_{i}' for i in range(Platoonsize + 1)])
    header += ',' + ','.join([f'Speed_{i}' for i in range(Platoonsize + 1)])
    header += ',' + ','.join([f'Accelerate_{i}' for i in range(Platoonsize + 1)])
    header += ',' + ','.join([f'Position_error_{i}' for i in range(Platoonsize + 1)])

    np.savetxt('.\\output\\vehicle_data-100.csv', data2, delimiter=',', header=header, comments='', fmt='%f')

    plt.figure(2)
    for i in range(Platoonsize + 1):
        if i == 0:
            label = 'leader'
        else:
            label = 'follower' + str(i)
        plt.plot(np.arange(len(position[i])), position[i],label = label)
    plt.xlabel('Time (s)')
    plt.ylabel('Location (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
    plt.savefig('.\\output\\Position time{}.png'.format(mkfile_time), dpi=300)  # 分别创建文件夹，分别储存命名图片

    plt.figure(3)
    for i in range(Platoonsize + 1):
        # if i == 0:
        #     label = 'leader'
        # else:
        #     label = 'follower' + str(i)
        # plt.plot(np.arange(len(speed[i])), speed[i],label = label)
        plt.plot(np.arange(len(speed[i])), speed[i])
    plt.xlabel('Time (s)')
    plt.ylabel('speed (m/s)')
    plt.grid(True)

    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
    plt.savefig('.\\output\\Speed time{}.png'.format(mkfile_time), dpi=300)  # 分别创建文件夹，分别储存命名图片

    plt.figure(4)
    for i in range(Platoonsize):
        plt.plot(np.arange(len(acclererate[i])), acclererate[i])
    plt.xlabel('Time (s)')
    plt.ylabel('acclerate (m/s)')
    plt.grid(True)

    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    plt.savefig('.\\output\\Acc time{}.png'.format(mkfile_time), dpi=300)



    plt.figure(5)
    for i in range(1,Platoonsize+1):
        plt.plot(np.arange(len(position_error[i])), position_error[i])
    plt.xlabel('Time (s)')
    plt.ylabel('position_error (m)')
    plt.grid(True)

    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    plt.savefig('.\\output\\position_error time{}.png'.format(mkfile_time), dpi=300)
    plt.figure(6)
    for i in range(1,Platoonsize+1):
        plt.plot(np.arange(len(speed_error[i])), speed_error[i])
    plt.xlabel('Time (s)')
    plt.ylabel('speed_error (m/s)')
    plt.grid(True)

    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    plt.savefig('.\\output\\speed_error time{}.png'.format(mkfile_time), dpi=300)
    plt.figure(7)
    for i in range(1,Platoonsize+1):
        plt.plot(np.arange(len(acc_error[i])), acc_error[i])
    plt.xlabel('Time (s)')
    plt.ylabel('acc_error (m/s)')
    plt.grid(True)

    plt.show()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    plt.savefig('.\\output\\acc_error time{}.png'.format(mkfile_time), dpi=300)






if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=100,
                        help="The number of neurons in hidden layers of the neural network")
    # parser.add_argument("--lr_a", type=float, default=0.0001, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=0.0002, help="Learning rate of critic")
    parser.add_argument("--lr_a", type=float, default=0.00008, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0001, help="Learning rate of critic")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    # parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    # parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    # parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_cuda", type=bool, default=False, help="cuda")

    args = parser.parse_args()


    main(args, seed=10)
