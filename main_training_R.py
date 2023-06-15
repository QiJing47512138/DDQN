import simpy
import sys
sys.path 
import numpy as np
import torch
from tabulate import tabulate
import os

import agent_machine
import agent_workcenter
import brain_workcenter_R
import job_creation
import breakdown_creation
import validation_S
import matplotlib.pyplot as plt
import matplotlib.animation as animation




"""
THIS IS THE MODULE FOR ROUTING AGENT TRAINING
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no,**kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        # 创建环境实例 指定模拟范围
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []

        self.epoch_rewards = []

        # create a list called rewards


        self.rewards = []  
        self.iterations = []  

        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        # '[1,50]' : Creates the range of resources required for the work task
        #'3' : The processing time required to create a new task
        # '0.8' : represents the ratio between the time required for the work and the time required to complete the work
        self.job_creator = job_creation.creation\
        (self.env, self.span, self.m_list, self.wc_list, [1,50], 3, 0.8, random_seed = True)
        self.job_creator.output()

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set up the brains for workcenters'''
        self.routing_brain = brain_workcenter_R.routing_brain(self.env, self.job_creator, \
            self.m_list, self.wc_list, self.span/5, self.span)

        

        '''STEP 6: run the simulaiton'''
        env.run()
        self.routing_brain.check_parameter()

    def reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
    # right half, showing the record of rewards
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.rt_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.rt_reward_record).transpose()[1]
        #print(time, rewards)
        reward_record.scatter(time, rewards, s=1,color='g', alpha=0.3, zorder=3)
        reward_record.set_xlim(0,self.span)
        reward_record.set_ylim(-1.1,1.1)
        xtick_interval = 2000
        reward_record.set_xticks(np.arange(0,self.span+1,xtick_interval))
        reward_record.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        reward_record.set_yticks(np.arange(-1, 1.1, 0.1))
        reward_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        reward_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        # moving average
        x = 50
        print(len(time))
        reward_record.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        reward_record.legend()
        plt.show()
        # save the figure if required
        fig.subplots_adjust(top=0.5, bottom=0.5, right=0.9)
        if 'save' in kwargs and kwargs['save']:
            fig.savefig(os.path.join("experiment_result", "RA_reward_{}wc_{}m.png".format(len(self.job_creator.wc_list),len(self.m_list))), dpi=500, bbox_inches='tight')
        return

    def calc_reward(self):
        # 计算当前 epoch 中的平均奖励值
        self.epoch_rewards = list(self.epoch_rewards)
        mean_reward = np.mean(self.epoch_rewards)

        # 将平均奖励值添加到奖励列表中
        self.rewards.append(mean_reward)

        #清空当前 epoch 的奖励列表，为下一轮 epoch 做准备
        self.epoch_rewards = []
        return mean_reward

    def plot_training_rewards(self,iterations, rewards):
        plt.plot(iterations, rewards)
        plt.xlabel('iterations')
        plt.ylabel('Average Reward')
        plt.title('Reward over iterations')
        
        
    #plot_training_rewards(iterations, rewards)
    
    def train_generator(self):
        self.epoch_rewards = list(self.epoch_rewards)
        for value in self.routing_brain.train():
            self.epoch_rewards.append(value)
            yield value
        

    def train_rewards(self, num_epochs, num_iterations):
        self.epoch_rewards = list(self.epoch_rewards)
        self.rewards = []
        self.iterations = []
        self.env.process(self.train_generator())
    
        # model train
        for i in range(num_epochs):
            self.epoch_rewards = []
            for j in range(num_iterations):
                # 执行训练及相关操作
                #reward = self.train_generator()
                rewards_generator = self.train_generator()
                for reward in rewards_generator:
                    self.epoch_rewards.append(reward)

                #self.epoch_rewards.append(reward)

            self.calc_reward()
        
            reward = self.calc_reward()
            self.epoch_rewards.append(reward)
            mean_reward = np.mean(self.epoch_rewards)
           
            self.rewards.append((i + 1,mean_reward))
            
            self.iterations.append(i + 1)

        self.plot_training_rewards(self.iterations,self.rewards) 

        #等会  plt.show()
            

        # mean_reward = np.mean(list(self.epoch_rewards))
        # mean_reward = np.mean([x for x in self.epoch_rewards if isinstance(x, int)])
        
        
        #return mean_reward
        return self.rewards
   

# create the environment instance for simulation
env = simpy.Environment()
# create the shop floor instance
span = 100000
m_no = 6
wc_no = 3
num_epochs = 10
num_iterations = 100


spf = shopfloor(env, span, m_no, wc_no)
#mean_reward = spf.train_rewards (num_epochs, num_iterations)
#rewards = spf.train_rewards(num_epochs, num_iterations)


huatu = spf.reward_record_output()