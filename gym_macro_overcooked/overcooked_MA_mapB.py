import numpy as np
from queue import PriorityQueue
from gym import spaces
from .items import Tomato, Onion, Lettuce, Plate, Knife, Delivery, Agent, Food, DirtyPlate, BadLettuce
from .overcooked import Overcooked
from .mac_agent import MacAgent
import random
from collections import deque
import time
import torch
import torch.nn as nn


DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "dirtyplate", "badlettuce"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8, "dirtyplate": 9, "badlettuce": 10}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
ACTIONIDX = {"right": 0, "down": 1, "left": 2, "up": 3, "stay": 4}
PRIMITIVEACTION =["right", "down", "left", "up", "stay"]

class AStarAgent(object):
    def __init__(self, x, y, g, dis, action, history_action, pass_agent):

        """
        Parameters
        ----------
        x : int
            X position of the agent.
        y : int
            Y position of the agent.
        g : int 
            Cost of the path from the start node to n.
        dis : int
            Distance of the current path.
            g + h
        pass_agent : int
            Whether there is other agent in the path.
        """

        self.x = x
        self.y = y
        self.g = g
        self.dis = dis
        self.action = action
        self.history_action = history_action
        self.pass_agent = pass_agent

    def __lt__(self, other):
        if self.dis != other.dis:
            return self.dis <= other.dis
        else:
            return self.pass_agent <= other.pass_agent


# 里面多了一些从macro到primitive action的转化，以及一些函数的重写。具体的如何step，还是转化为执行low level action。此外，还制作了一个Wrapper，让gym env可以调用
class Overcooked_MA_mapB(Overcooked):

    """
    Overcooked Domain Description
    ------------------------------
    ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
    map_type = ["A", "B", "C"]

    Only macro-action is available in this env.
    Macro-actions in map A:
    ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "right", "down", "left", "up"]
    Macro-actions in map B/C:
    ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter", "right", "down", "left", "up"]
    
    1) Agent is allowed to pick up/put down food/plate on the counter;
    2) Agent is allowed to chop food into pieces if the food is on the cutting board counter;
    3) Agent is allowed to deliver food to the delivery counter;
    4) Only unchopped food is allowed to be chopped;
    """
        
    def __init__(self, grid_dim, task, rewardList, map_type = "A", n_agent = 2, obs_radius = 2, mode = "vector", debug = False):

        """
        Parameters
        ----------
        gird_dim : tuple(int, int)
            The size of the grid world([7, 7]/[9, 9]).
        task : int
            The index of the target recipe.
        rewardList : dictionary
            The list of the reward.
            e.g rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
        map_type : str 
            The type of the map(A/B/C).
        n_agent: int
            The number of the agents.
        obs_radius: int
            The radius of the agents.
        mode: string
            The type of the observation(vector/image).
        debug : bool
            Whehter print the debug information.
        """

        super().__init__(grid_dim, task, rewardList, map_type, n_agent, obs_radius, mode, debug)
        self.macroAgent = []
        self._createMacroAgents()
        self.macroActionItemList = []
        self._createMacroActionItemList()

        self.macroActionName = ["stay", "get lettuce 1", "get lettuce 2", "get badlettuce", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver 1", "deliver 2", "chop", "go to counter", "right", "down", "left", "up"]

        self.action_space = spaces.Discrete(len(self.macroActionName))


        self.target_counter_x = 0
        self.target_counter_y = 0

        self.target_plate_destination_x = 0
        self.target_plate_destination_y = 0

    def _createMacroAgents(self):
        for agent in self.agent:
            self.macroAgent.append(MacAgent())

    def _createMacroActionItemList(self):
        self.macroActionItemList = []
        for key in self.itemDic:
            if key != "agent":
                self.macroActionItemList += self.itemDic[key]

    def macro_action_sample(self):
        macro_actions = []
        for agent in self.agent:
            macro_actions.append(random.randint(0, self.action_space.n - 1))
        return macro_actions     

    def build_agents(self):
        raise

    def build_macro_actions(self):
        raise


    # 根据macro action来找到要去的位置
    def _findPOitem(self, agent, macro_action):
    
        """
        Parameters
        ----------
        agent : Agent
        macro_action: int

        Returns
        -------
        x : int
            X position of the item in the observation of the agent.
        y : int
            Y position of the item in the observation of the agent.
        """
        
        deliver_idx = self.macroActionName.index("deliver 1") 
        if macro_action < deliver_idx:
            idx = (macro_action - 1) * 3
        else:
            idx = (macro_action - 1) * 2 + (deliver_idx - 1)

        return int(agent.obs[idx] * self.xlen), int(agent.obs[idx + 1] * self.ylen)
    


    def reset(self):
                
        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """

        super().reset()
        for agent in self.macroAgent:
            agent.reset()
        return self._get_macro_obs()

    def run(self, macro_actions):

        """
        Parameters
        ----------
        macro_actions: list
            macro_action for each agent

        Returns
        -------
        macro_obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        print('=====================使用了RUN')
        actions = self._computeLowLevelActions(macro_actions)
        
        obs, rewards, terminate, info = self.step(actions)

        self._checkCollision(info)
        cur_mac = self._collectCurMacroActions()
        mac_done = self._computeMacroActionDone()

        self._createMacroActionItemList()

        info = {'cur_mac': cur_mac, 'mac_done': mac_done}
        return  self._get_macro_obs(), rewards, terminate, info

    def _checkCollision(self, info):
        for idx in info["collision"]:
            self.macroAgent[idx].cur_macro_action_done = True



    def _checkMacroActionDone(self):
        # loop each agent
        for idx, agent in enumerate(self.agent):
            if not self.macroAgent[idx].cur_macro_action_done:
                macro_action = self.macroAgent[idx].cur_macro_action
                if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion"]:
                    target_x, target_y = self._findPOitem(agent, macro_action)

                    macroAction2ItemName = {"get tomato": "tomato", "get lettuce 1": "lettuce", "get lettuce 2": "lettuce", "get badlettuce": "badlettuce", "get onion": "onion"}
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        for knife in self.knife:
                            if knife.x == target_x and knife.y == target_y:
                                food = self._findItem(target_x, target_y, macroAction2ItemName[self.macroActionName[macro_action]])
                                if not food.chopped:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    break
                elif self.macroActionName[macro_action] in ["deliver 1", "deliver 2"] and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif self.mapType in ["B", "C"] and self.macroActionName[macro_action] == "go to counter" and not agent.holding:
                    target_x = 0
                    target_y = int(self.ylen // 2)
                    findEmptyCounter = False
                    for i in self.counterSequence:
                        if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                            target_x = i
                            findEmptyCounter = True
                            break
                    if findEmptyCounter:
                        if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                            self.macroAgent[idx].cur_macro_action_done = True
                    else:
                        self.macroAgent[idx].cur_macro_action_done = True

                if self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]\
                    or self.macroActionName[macro_action] in ["get plate 1", "get plate 2"]:
                        target_x, target_y = self._findPOitem(agent, macro_action)
                        macroAction2Item = {"get tomato": self.tomato[0], "get lettuce": self.lettuce[0], "get onion": self.onion[0], "get plate 1": self.plate[0], "get plate 2": self.plate[1]}
                        item = macroAction2Item[self.macroActionName[macro_action]]
                        if target_x != item.x or target_y != item.y:
                            self.macroAgent[idx].cur_macro_action_done = True


    def _computeLowLevelActions(self, macro_actions):
    # def _computeLowLevelActions(self, macro_actions):
        """
        Parameters
        ----------
        macro_actions : int | List[..]
            The discrete macro-actions index for the agents. 

        Returns
        -------
        primitive_actions : int | List[..]
            The discrete primitive-actions index for the agents. 
        """
        real_execute_macro_actions = []

        primitive_actions = []
        
        counter_y_in_this_map = 7
        
        # loop each agent
        for idx, agent in enumerate(self.agent):



            # print("done " + str(idx), self.macroAgent[idx].cur_macro_action_done)

            """下面是判断是否done的操作，我试着允许打断，允许打断的话，那就先注释掉"""


            # if self.macroAgent[idx].cur_macro_action_done:
            #     self.macroAgent[idx].cur_macro_action = macro_actions[idx]
            #     macro_action = macro_actions[idx]
            #     self.macroAgent[idx].cur_macro_action_done = False
            # else:
            #     macro_action = self.macroAgent[idx].cur_macro_action
            




            if self.macroAgent[idx].cur_macro_action_done:

                if self.macroActionName[macro_actions[idx]] == "get plate 1" or self.macroActionName[macro_actions[idx]] == "get plate 2":

                    tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.plate[0].x, self.plate[0].y)
                    tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.plate[1].x, self.plate[1].y)

                    if tolettuce1 == -1 and tolettuce2 != -1:
                        which_lettuce = 5
                    
                    if tolettuce2 == -1 and tolettuce1 != -1:
                        which_lettuce = 4

                    if tolettuce1 == -1 and tolettuce2 == -1:
                        which_lettuce = 0


                    """注意,这里与mapA不同,mapA是<=和>,这里是<和>=,为了避免队友递盘子导致误拿"""
                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 < tolettuce2:
                        which_lettuce = 4
                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 >= tolettuce2:
                        which_lettuce = 5
                        
                    target_x, target_y = self._findPOitem(agent, 4)
                    if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                        which_lettuce = 5
                    target_x, target_y = self._findPOitem(agent, 5)
                    if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                        which_lettuce = 4

                    if idx == 1:
                        target_x1, target_y1 = self._findPOitem(agent, 4)
                        target_x2, target_y2 = self._findPOitem(agent, 5)

                        # print(target_x1, target_x2)

                        if target_y1 == counter_y_in_this_map and target_y2 != counter_y_in_this_map:
                            which_lettuce = 5
                        if target_y2 == counter_y_in_this_map and target_y1 != counter_y_in_this_map:
                            which_lettuce = 4
                        if target_y1 == counter_y_in_this_map and target_y2 == counter_y_in_this_map:
                            # print('都在counter上')
                            which_lettuce = 0
                        if target_y1 != counter_y_in_this_map and target_y2 != counter_y_in_this_map:
                            tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.plate[0].x, self.plate[0].y)
                            tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.plate[1].x, self.plate[1].y)

                            if tolettuce1 == -1 and tolettuce2 != -1:
                                which_lettuce = 5
                            
                            if tolettuce2 == -1 and tolettuce1 != -1:
                                which_lettuce = 4

                            if tolettuce1 == -1 and tolettuce2 == -1:
                                which_lettuce = 0

                            if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 <= tolettuce2:
                                which_lettuce = 4
                            if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 > tolettuce2:
                                which_lettuce = 5
                                
                            target_x, target_y = self._findPOitem(agent, 4)
                            if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                                which_lettuce = 5
                            target_x, target_y = self._findPOitem(agent, 5)
                            if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                                which_lettuce = 4




                    self.macroAgent[idx].cur_macro_action = which_lettuce
                    macro_action = which_lettuce
                    self.macroAgent[idx].cur_macro_action_done = False


                elif self.macroActionName[macro_actions[idx]] == "go to knife 1" or self.macroActionName[macro_actions[idx]] == "go to knife 2":

                    tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.knife[0].x, self.knife[0].y)
                    tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.knife[1].x, self.knife[1].y)

                    if tolettuce1 == -1 and tolettuce2 != -1:
                        which_lettuce = 7
                    
                    if tolettuce2 == -1 and tolettuce1 != -1:
                        which_lettuce = 6

                    if tolettuce1 == -1 and tolettuce2 == -1:
                        which_lettuce = 0

                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 <= tolettuce2:
                        which_lettuce = 6
                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 > tolettuce2:
                        which_lettuce = 7
                        
                    target_x, target_y = self._findPOitem(agent, 6)
                    if self.knife[0].holding and agent.holding:
                        which_lettuce = 7
                    target_x, target_y = self._findPOitem(agent, 7)
                    if self.knife[1].holding and agent.holding:
                        which_lettuce = 6


                    self.macroAgent[idx].cur_macro_action = which_lettuce
                    macro_action = which_lettuce
                    self.macroAgent[idx].cur_macro_action_done = False



                elif self.macroActionName[macro_actions[idx]] == "deliver 1" or self.macroActionName[macro_actions[idx]] == "deliver 2":

                    tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.delivery[0].x, self.delivery[0].y)
                    tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.delivery[1].x, self.delivery[1].y)

                    # print(tolettuce1, tolettuce2)

                    if tolettuce1 == -1 and tolettuce2 != -1:
                        which_lettuce = 9
                    
                    if tolettuce2 == -1 and tolettuce1 != -1:
                        which_lettuce = 8

                    if tolettuce1 == -1 and tolettuce2 == -1:
                        which_lettuce = 0

                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 <= tolettuce2:
                        which_lettuce = 8
                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 > tolettuce2:
                        which_lettuce = 9



                    if idx == 1 and isinstance(agent.holding, Lettuce):
                        which_lettuce = 0



                    self.macroAgent[idx].cur_macro_action = which_lettuce
                    macro_action = which_lettuce
                    self.macroAgent[idx].cur_macro_action_done = False


                elif self.macroActionName[macro_actions[idx]] == "get lettuce 1" or self.macroActionName[macro_actions[idx]] == "get lettuce 2":
                    which_lettuce = None

                    tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.lettuce[0].x, self.lettuce[0].y)
                    tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.lettuce[1].x, self.lettuce[1].y)
                    
                    if tolettuce1 == -1 and tolettuce2 != -1:
                        which_lettuce = 2
                    
                    if tolettuce2 == -1 and tolettuce1 != -1:
                        which_lettuce = 1

                    if tolettuce1 == -1 and tolettuce2 == -1:
                        which_lettuce = 0

                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 <= tolettuce2:
                        which_lettuce = 1
                    if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 > tolettuce2:
                        which_lettuce = 2


                    target_x, target_y = self._findPOitem(agent, 1)
                    if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                        which_lettuce = 2
                    target_x, target_y = self._findPOitem(agent, 2)
                    if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                        which_lettuce = 1

                    if idx == 1:
                        target_x1, target_y1 = self._findPOitem(agent, 1)
                        target_x2, target_y2 = self._findPOitem(agent, 2)

                        if target_y1 == counter_y_in_this_map and target_y2 != counter_y_in_this_map:
                            # print('情况1')
                            which_lettuce = 2
                        if target_y2 == counter_y_in_this_map and target_y1 != counter_y_in_this_map:
                            # print('情况2')
                            which_lettuce = 1
                        if target_y2 == counter_y_in_this_map and target_y1 == counter_y_in_this_map:
                            # print('情况3')
                            which_lettuce = 0
                        if target_y2 != counter_y_in_this_map and target_y1 != counter_y_in_this_map:
                            # print('情况4')
                            # print(target_y2)
                            # print(counter_y_in_this_map)
                            tolettuce1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.lettuce[0].x, self.lettuce[0].y)
                            tolettuce2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, self.lettuce[1].x, self.lettuce[1].y)
                            
                            if tolettuce1 == -1 and tolettuce2 != -1:
                                which_lettuce = 2
                            
                            if tolettuce2 == -1 and tolettuce1 != -1:
                                which_lettuce = 1

                            if tolettuce1 == -1 and tolettuce2 == -1:
                                which_lettuce = 0

                            if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 <= tolettuce2:
                                which_lettuce = 1
                            if tolettuce1 != -1 and tolettuce2 != -1 and tolettuce1 > tolettuce2:
                                which_lettuce = 2


                            target_x, target_y = self._findPOitem(agent, 1)
                            if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                                which_lettuce = 2
                            target_x, target_y = self._findPOitem(agent, 2)
                            if ITEMNAME[agent.pomap[target_x][target_y]] == "agent":
                                which_lettuce = 1




                    self.macroAgent[idx].cur_macro_action = which_lettuce
                    macro_action = which_lettuce
                    self.macroAgent[idx].cur_macro_action_done = False
                else:

                    self.macroAgent[idx].cur_macro_action = macro_actions[idx]
                    macro_action = macro_actions[idx]
                    self.macroAgent[idx].cur_macro_action_done = False


            # 否则，还是继续执行上一个macro action
            else:
                # 不可打断
                macro_action = self.macroAgent[idx].cur_macro_action


            # print("agent " + str(idx), "macro_action: ", self.macroActionName[macro_action])


            real_execute_macro_actions.append(macro_action)

            # 先把primitive action设为4，stay
            primitive_action = ACTIONIDX["stay"]

            target_x, target_y = self._findPOitem(agent, macro_action)

            # print('目标位置是: ', target_x, target_y)

            if self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, target_x, target_y) == -1 and self._calDistance(agent.x, agent.y, target_x, target_y) == 2:
                # print('难道进入这里了？')
                self.macroAgent[idx].cur_macro_action_done = True
                primitive_action = ACTIONIDX["stay"]


            # 如果mac action是stay，那就可以直接标记为done了
            if macro_action == 0:
                self.macroAgent[idx].cur_macro_action_done = True
            # 如果mac action是chop
            elif self.macroActionName[macro_action] == "chop":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]] 
                    if new_name == "knife":
                        knife = self._findItem(new_x, new_y, new_name)
                        if isinstance(knife.holding, Food):
                            if not knife.holding.chopped:
                                primitive_action = action
                                self.macroAgent[idx].cur_chop_times += 1
                                if self.macroAgent[idx].cur_chop_times >= 1:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    self.macroAgent[idx].cur_chop_times = 0
                                break
                if primitive_action == ACTIONIDX["stay"]:
                    self.macroAgent[idx].cur_macro_action_done = True
            

            

            elif self.macroActionName[macro_action] == "go to counter":
                """这里我可以改一下go to counter"""
                # 遍历所有counter，计算每一个counter的可达性A，可达性B，距离A的距离，距离B的距离，这四个。
                # 然后找出可达性A和B都满足，且距离A和距离B之和最小的。
                # 然后找出里面距离A和距离B的差值的绝对值最小。


                # print('进入')
                

                distance_to_agent1 = []
                distance_to_agent2 = []
                canreach_agent1 = []
                canreach_agent2 = []

                counter_x = []
                counter_y = []

                # 这里每次更换地图都要改
                for x_i in [12, 13]:
                    for y_i in [7]:

                        counter_x.append(x_i)
                        counter_y.append(y_i)

                        distance_to_agent1.append(self._calDistance(x_i, y_i, agent.x, agent.y))
                        distance_to_agent2.append(self._calDistance(x_i, y_i, self.agent[1 - idx].x, self.agent[1 - idx].y))
                        canreach_agent1.append(self._navigate(agent, x_i, y_i))
                        canreach_agent2.append(self._navigate(self.agent[1 - idx], x_i, y_i))
               

                tocounter1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, 12, 7)
                tocounter2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, 13, 7)


                if tocounter1 == -1 and tocounter2 != -1:
                    best_index = 1
                
                if tocounter2 == -1 and tocounter1 != -1:
                    best_index = 0

                if tocounter1 == -1 and tocounter2 == -1:
                    best_index = None

                if tocounter1 != -1 and tocounter2 != -1 and tocounter1 <= tocounter2:
                    best_index = 0
                if tocounter1 != -1 and tocounter2 != -1 and tocounter1 > tocounter2:
                    best_index = 1

                # print('最好的counter: ', best_index)
                if best_index is None:
                    primitive_action = ACTIONIDX["stay"]
                    self.macroAgent[idx].cur_macro_action_done = True
                else:
                    if agent.holding and isinstance(agent.holding, Food) and ITEMNAME[agent.pomap[counter_x[best_index]][counter_y[best_index]]] != "counter":
                        self.target_counter_x = counter_x[1-best_index]
                        self.target_counter_y = counter_y[1-best_index]
                    else:
                        self.target_counter_x = counter_x[best_index]
                        self.target_counter_y = counter_y[best_index]


                    if self._calDistance(agent.x, agent.y, self.target_counter_x, self.target_counter_y) == 1 and not agent.holding:
                        primitive_action = ACTIONIDX["stay"]
                        self.macroAgent[idx].cur_macro_action_done = True
                    else:
                        primitive_action = self._navigate(agent, self.target_counter_x, self.target_counter_y)
                        if self._calDistance(agent.x, agent.y, self.target_counter_x, self.target_counter_y) == 1:
                            self.macroAgent[idx].cur_macro_action_done = True


            # 如果mac action是一些上下左右，就可以直接映射为primitive action了
            elif macro_action >= self.macroActionName.index("right"):
                self.macroAgent[idx].cur_macro_action_done = True
                action = macro_action - self.macroActionName.index("right")
                new_x = agent.x + DIRECTION[action][0]
                new_y = agent.y + DIRECTION[action][1]
                if ITEMNAME[agent.pomap[new_x][new_y]] == "space":
                    primitive_action = action
                else:
                    primitive_action = ACTIONIDX["stay"]



            else:
                """
                # 当宏动作未被直接处理（例如 "deliver"），代码会进入以下部分：
                """

                target_x, target_y = self._findPOitem(agent, macro_action)

                inPlate = False

                if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion"]:
                    # 如果目标在视野范围之内
                    if (target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) \
                        or self.obs_radius == 0:
                        for plate in self.plate:
                            if plate.x == target_x and plate.y == target_y:
                                # print('啥意思啊，会出现这个问题吗？')
                                primitive_action = ACTIONIDX["stay"]
                                self.macroAgent[idx].cur_macro_action_done = True
                                inPlate = True
                                break
                        for plate in self.dirtyplate:
                            if plate.x == target_x and plate.y == target_y:
                                primitive_action = ACTIONIDX["stay"]
                                self.macroAgent[idx].cur_macro_action_done = True
                                inPlate = True
                                break
                    # print('inPlate: ', inPlate)
                if inPlate:
                    primitive_actions.append(primitive_action)
                    continue
            

                elif ITEMNAME[agent.pomap[target_x][target_y]] == "agent" \
                    and ((target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) or self.obs_radius == 0):
                    self.macroAgent[idx].cur_macro_action_done = True


                else:
                    if agent.holding and isinstance(agent.holding, Food) and (self.macroActionName[macro_action] == "get lettuce 1" or self.macroActionName[macro_action] == "get lettuce 2" or self.macroActionName[macro_action] == "get badlettuce"):
                        self.macroAgent[idx].cur_macro_action_done = True
                        primitive_action = ACTIONIDX["stay"]
                    else:
                        primitive_action = self._navigate(agent, target_x, target_y)
                        if primitive_action == ACTIONIDX["stay"]:
                            self.macroAgent[idx].cur_macro_action_done = True

                        # 如果拿着盘子去切菜板，除非切菜板上是切好的蔬菜，否则一律取消
                        if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and (isinstance(agent.holding, Plate) or isinstance(agent.holding, DirtyPlate)):

                            target_x, target_y = self._findPOitem(agent, macro_action)

                            knife_item_here = self._findItem(target_x, target_y, "knife")
                            if not (knife_item_here.holding and (isinstance(knife_item_here.holding, Lettuce) or isinstance(knife_item_here.holding, BadLettuce)) and knife_item_here.holding.chopped):
                                # self.macroAgent[idx].cur_macro_action_done = True

                                # 如果拿着盘子去不合理的切菜板了，就要把盘子放下

                                # 1. 收集所有合法 counter 的位置
                                counter_positions = []
                                distance_to_knife = []

                                for x_i in range(self.xlen):
                                    for y_i in range(self.ylen):
                                        if ITEMNAME[agent.pomap[x_i][y_i]] == "counter":
                                            counter_positions.append((x_i, y_i))
                                            distance_to_knife.append(self._calDistance(x_i, y_i, target_x, target_y))

                                # 2. 使用 find_nearest_reachable_target 找到最近的合法 counter
                                agent_x, agent_y = agent.x, agent.y
                                counter_index = self.find_nearest_reachable_target(agent.pomap, agent_x, agent_y, counter_positions)

                                # print(counter_positions[counter_index][0], counter_positions[counter_index][1])

                                primitive_action = self._navigate(agent, counter_positions[counter_index][0], counter_positions[counter_index][1])


                                if self._calDistance(agent.x, agent.y, counter_positions[counter_index][0], counter_positions[counter_index][1]) == 1:
                                    self.macroAgent[idx].cur_macro_action_done = True



                        """新增1：拿着切好的菜去knife"""
                        if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and (isinstance(agent.holding, Food) and agent.holding.chopped):
                            self.macroAgent[idx].cur_macro_action_done = True
                            primitive_action = ACTIONIDX["stay"]

                        """新增2：拿着盘子去取还没切的菜"""
                        # if self.macroActionName[macro_action] == "get lettuce 1" and isinstance(agent.holding, Plate) and not self.lettuce[0].chopped:
                        #     self.macroAgent[idx].cur_macro_action_done = True
                        #     primitive_action = ACTIONIDX["stay"]

                        # if self.macroActionName[macro_action] == "get lettuce 2" and isinstance(agent.holding, Plate) and not self.lettuce[1].chopped:
                        #     self.macroAgent[idx].cur_macro_action_done = True
                        #     primitive_action = ACTIONIDX["stay"]

                        # if self.macroActionName[macro_action] == "get badlettuce" and isinstance(agent.holding, Plate) and not self.badlettuce[0].chopped:
                        #     self.macroAgent[idx].cur_macro_action_done = True
                        #     primitive_action = ACTIONIDX["stay"]


                        if (self.macroActionName[macro_action] == "get lettuce 1" and isinstance(agent.holding, Plate) and not self.lettuce[0].chopped) or (self.macroActionName[macro_action] == "get lettuce 2" and isinstance(agent.holding, Plate) and not self.lettuce[1].chopped) or (self.macroActionName[macro_action] == "get badlettuce" and isinstance(agent.holding, Plate) and not self.badlettuce[0].chopped):



                            target_x, target_y = self._findPOitem(agent, 6)

                            knife_item_here = self._findItem(target_x, target_y, "knife")
                            if not (knife_item_here.holding and (isinstance(knife_item_here.holding, Lettuce) or isinstance(knife_item_here.holding, BadLettuce)) and knife_item_here.holding.chopped):
                                # self.macroAgent[idx].cur_macro_action_done = True

                                # 如果拿着盘子去不合理的切菜板了，就要把盘子放下

                                # 1. 收集所有合法 counter 的位置
                                counter_positions = []
                                distance_to_knife = []

                                for x_i in range(self.xlen):
                                    for y_i in range(self.ylen):
                                        if ITEMNAME[agent.pomap[x_i][y_i]] == "counter":
                                            counter_positions.append((x_i, y_i))
                                            distance_to_knife.append(self._calDistance(x_i, y_i, target_x, target_y))

                                # 2. 使用 find_nearest_reachable_target 找到最近的合法 counter
                                agent_x, agent_y = agent.x, agent.y
                                counter_index = self.find_nearest_reachable_target(agent.pomap, agent_x, agent_y, counter_positions)

                                # print(counter_positions[counter_index][0], counter_positions[counter_index][1])

                                primitive_action = self._navigate(agent, counter_positions[counter_index][0], counter_positions[counter_index][1])


                                if self._calDistance(agent.x, agent.y, counter_positions[counter_index][0], counter_positions[counter_index][1]) == 1:
                                    self.macroAgent[idx].cur_macro_action_done = True



                        """新增3：拿着盘子去取盘子"""
                        if self.macroActionName[macro_action] in ["get plate 1", "get plate 2"] and isinstance(agent.holding, Plate):
                            self.macroAgent[idx].cur_macro_action_done = True
                            primitive_action = ACTIONIDX["stay"]





                        if self.macroActionName[macro_action] in ["get plate 1", "get plate 2", "get dirty plate"]:
                            if isinstance(agent.holding, Food) and not agent.holding.chopped:
                                self.macroAgent[idx].cur_macro_action_done = True


                        if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:

                            self.macroAgent[idx].cur_macro_action_done = True
                            if self.macroActionName[macro_action] in ["get plate 1", "get plate 2", "get dirty plate"] and agent.holding:
                                if isinstance(agent.holding, Food):
                                    if agent.holding.chopped:
                                        self.macroAgent[idx].cur_macro_action_done = False
                                    else:
                                        primitive_action = ACTIONIDX["stay"]
                            
                            if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and not agent.holding:
                                primitive_action = ACTIONIDX["stay"]

                            if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion"]:
                                    for knife in self.knife:
                                        if knife.x == target_x and knife.y == target_y:
                                            if isinstance(knife.holding, Food):
                                                if not knife.holding.chopped:
                                                    primitive_action = ACTIONIDX["stay"]
                                                    break                           
                            
                            # 如果目标位置发生了移动
                            if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion", "get plate 1", "get plate 2", "get dirty plate"]:
                                macroAction2Item = {}
                                if self.macroActionName[macro_action] == "get tomato 1":
                                    macroAction2Item["get tomato 1"] = self.tomato[0]
                                elif self.macroActionName[macro_action] == "get tomato 2":
                                    macroAction2Item["get tomato 2"] = self.tomato[1]
                                elif self.macroActionName[macro_action] == "get lettuce 1":
                                    macroAction2Item["get lettuce 1"] = self.lettuce[0]
                                elif self.macroActionName[macro_action] == "get lettuce 2":
                                    macroAction2Item["get lettuce 2"] = self.lettuce[1]
                                elif self.macroActionName[macro_action] == "get badlettuce":
                                    macroAction2Item["get badlettuce"] = self.badlettuce[0]
                                elif self.macroActionName[macro_action] == "get onion":
                                    macroAction2Item["get onion"] = self.onion[0]
                                elif self.macroActionName[macro_action] == "get plate 1":
                                    if isinstance(agent.holding, Food) and not agent.holding.chopped:
                                        self.macroAgent[idx].cur_macro_action_done = True
                                    macroAction2Item["get plate 1"] = self.plate[0]
                                elif self.macroActionName[macro_action] == "get plate 2":
                                    if isinstance(agent.holding, Food) and not agent.holding.chopped:
                                        self.macroAgent[idx].cur_macro_action_done = True
                                    macroAction2Item["get plate 2"] = self.plate[1]
                                elif self.macroActionName[macro_action] == "get dirty plate":
                                    if isinstance(agent.holding, Food) and not agent.holding.chopped:
                                        self.macroAgent[idx].cur_macro_action_done = True
                                    macroAction2Item["get dirty plate"] = self.dirtyplate[0]

                                item = macroAction2Item[self.macroActionName[macro_action]]
                                if target_x != item.x or target_y != item.y:
                                    primitive_action = ACTIONIDX["stay"]

            # 返回的其实只是两个agent下一个step要做的primitive action，而不是一个primitive action序列
            primitive_actions.append(primitive_action)
        return primitive_actions, real_execute_macro_actions






    def _computeLowLevelActions_boltzmaanlowlevel(self, macro_actions):
    # def _computeLowLevelActions(self, macro_actions):
        """
        Parameters
        ----------
        macro_actions : int | List[..]
            The discrete macro-actions index for the agents. 

        Returns
        -------
        primitive_actions : int | List[..]
            The discrete primitive-actions index for the agents. 
        """
        real_execute_macro_actions = []

        primitive_actions = []
        # loop each agent
        for idx, agent in enumerate(self.agent):

            macro_action = macro_actions[idx]


            # print("实际执行的agent " + str(idx), "macro_action: ", self.macroActionName[macro_action])


            real_execute_macro_actions.append(macro_action)

            # 先把primitive action设为4，stay
            primitive_action = ACTIONIDX["stay"]

            # target_x, target_y = self._findPOitem(agent, macro_action)
            # if self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, target_x, target_y) == -1 and self._calDistance(agent.x, agent.y, target_x, target_y) == 2:
            #     primitive_action = ACTIONIDX["stay"]


            if self.macroActionName[macro_action] == "chop":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]] 
                    if new_name == "knife":
                        knife = self._findItem(new_x, new_y, new_name)
                        if isinstance(knife.holding, Food):
                            if not knife.holding.chopped:
                                primitive_action = action
                                self.macroAgent[idx].cur_chop_times += 1
                                if self.macroAgent[idx].cur_chop_times >= 1:
                                    self.macroAgent[idx].cur_chop_times = 0
                                break
          

            

            elif self.macroActionName[macro_action] == "go to counter":
                """这里我可以改一下go to counter"""
                # 遍历所有counter，计算每一个counter的可达性A，可达性B，距离A的距离，距离B的距离，这四个。
                # 然后找出可达性A和B都满足，且距离A和距离B之和最小的。
                # 然后找出里面距离A和距离B的差值的绝对值最小。
                

                distance_to_agent1 = []
                distance_to_agent2 = []
                canreach_agent1 = []
                canreach_agent2 = []

                counter_x = []
                counter_y = []

                # 这里每次更换地图都要改
                for x_i in [12, 13]:
                    for y_i in [7]:

                        counter_x.append(x_i)
                        counter_y.append(y_i)

                        distance_to_agent1.append(self._calDistance(x_i, y_i, agent.x, agent.y))
                        distance_to_agent2.append(self._calDistance(x_i, y_i, self.agent[1 - idx].x, self.agent[1 - idx].y))
                        canreach_agent1.append(self._navigate(agent, x_i, y_i))
                        canreach_agent2.append(self._navigate(self.agent[1 - idx], x_i, y_i))
               

                tocounter1 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, 12, 7)
                tocounter2 = self.shortest_path_through_zeros(agent.pomap, agent.x, agent.y, 13, 7)


                if tocounter1 == -1 and tocounter2 != -1:
                    best_index = 1
                
                if tocounter2 == -1 and tocounter1 != -1:
                    best_index = 0

                if tocounter1 == -1 and tocounter2 == -1:
                    best_index = None

                if tocounter1 != -1 and tocounter2 != -1 and tocounter1 <= tocounter2:
                    best_index = 0
                if tocounter1 != -1 and tocounter2 != -1 and tocounter1 > tocounter2:
                    best_index = 1


                if best_index is None:
                    primitive_action = ACTIONIDX["stay"]
                else:
                    if agent.holding and isinstance(agent.holding, Food) and ITEMNAME[agent.pomap[counter_x[best_index]][counter_y[best_index]]] != "counter":
                        self.target_counter_x = counter_x[1-best_index]
                        self.target_counter_y = counter_y[1-best_index]
                    else:
                        self.target_counter_x = counter_x[best_index]
                        self.target_counter_y = counter_y[best_index]



                    if self._calDistance(agent.x, agent.y, self.target_counter_x, self.target_counter_y) == 1 and not agent.holding:
                        primitive_action = ACTIONIDX["stay"]
                    else:
                        # if idx == 1 and partner_ability == 0:
                        #     primitive_action = self._navigate_softmin_one_step(agent, self.target_counter_x, self.target_counter_y, beta=2.0)
                        # else:
                        primitive_action = self._navigate(agent, self.target_counter_x, self.target_counter_y)
                            


            # 如果mac action是一些上下左右，就可以直接映射为primitive action了
            elif macro_action >= self.macroActionName.index("right"):
                action = macro_action - self.macroActionName.index("right")
                new_x = agent.x + DIRECTION[action][0]
                new_y = agent.y + DIRECTION[action][1]
                if ITEMNAME[agent.pomap[new_x][new_y]] == "space":
                    primitive_action = action
                else:
                    primitive_action = ACTIONIDX["stay"]



            else:
                """
                # 当宏动作未被直接处理（例如 "deliver"），代码会进入以下部分：
                """

                target_x, target_y = self._findPOitem(agent, macro_action)

                inPlate = False

                if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion"]:
                    # 如果目标在视野范围之内
                    if (target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) \
                        or self.obs_radius == 0:
                        for plate in self.plate:
                            if plate.x == target_x and plate.y == target_y:
                                # print('啥意思啊，会出现这个问题吗？')
                                primitive_action = ACTIONIDX["stay"]
                                inPlate = True
                                break
                        for plate in self.dirtyplate:
                            if plate.x == target_x and plate.y == target_y:
                                primitive_action = ACTIONIDX["stay"]
                                inPlate = True
                                break
                    # print('inPlate: ', inPlate)
                if inPlate:
                    primitive_actions.append(primitive_action)
                    continue
            




                else:
                    if agent.holding and isinstance(agent.holding, Food) and (self.macroActionName[macro_action] == "get lettuce 1" or self.macroActionName[macro_action] == "get lettuce 2" or self.macroActionName[macro_action] == "get badlettuce"):

                        primitive_action = ACTIONIDX["stay"]
                    else:


                        # if idx == 1 and partner_ability == 0:
                        #     primitive_action = self._navigate_softmin_one_step(agent, target_x, target_y, beta=2.0)
                            
                        # else:
                        primitive_action = self._navigate(agent, target_x, target_y)



                        # 如果拿着盘子去切菜板，除非切菜板上是切好的蔬菜，否则一律取消
                        if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and (isinstance(agent.holding, Plate) or isinstance(agent.holding, DirtyPlate)):

                            target_x, target_y = self._findPOitem(agent, macro_action)

                            knife_item_here = self._findItem(target_x, target_y, "knife")
                            if not (knife_item_here.holding and (isinstance(knife_item_here.holding, Lettuce) or isinstance(knife_item_here.holding, BadLettuce)) and knife_item_here.holding.chopped):
                                # self.macroAgent[idx].cur_macro_action_done = True

                                # 如果拿着盘子去不合理的切菜板了，就要把盘子放下

                                # 1. 收集所有合法 counter 的位置
                                counter_positions = []
                                distance_to_knife = []

                                for x_i in range(self.xlen):
                                    for y_i in range(self.ylen):
                                        if ITEMNAME[agent.pomap[x_i][y_i]] == "counter":
                                            counter_positions.append((x_i, y_i))
                                            distance_to_knife.append(self._calDistance(x_i, y_i, target_x, target_y))

                                # 2. 使用 find_nearest_reachable_target 找到最近的合法 counter
                                agent_x, agent_y = agent.x, agent.y
                                counter_index = self.find_nearest_reachable_target(agent.pomap, agent_x, agent_y, counter_positions)

                                # print(counter_positions[counter_index][0], counter_positions[counter_index][1])

                                primitive_action = self._navigate(agent, counter_positions[counter_index][0], counter_positions[counter_index][1])



                        """新增1：拿着切好的菜去knife"""
                        if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and (isinstance(agent.holding, Food) and agent.holding.chopped):
                            primitive_action = ACTIONIDX["stay"]

                        """新增2：拿着盘子去取还没切的菜"""
                        # if self.macroActionName[macro_action] == "get lettuce 1" and isinstance(agent.holding, Plate) and not self.lettuce[0].chopped:
                        #     primitive_action = ACTIONIDX["stay"]

                        # if self.macroActionName[macro_action] == "get lettuce 2" and isinstance(agent.holding, Plate) and not self.lettuce[1].chopped:
                        #     primitive_action = ACTIONIDX["stay"]

                        # if self.macroActionName[macro_action] == "get badlettuce" and isinstance(agent.holding, Plate) and not self.badlettuce[0].chopped:
                        #     primitive_action = ACTIONIDX["stay"]


                        if (self.macroActionName[macro_action] == "get lettuce 1" and isinstance(agent.holding, Plate) and not self.lettuce[0].chopped) or (self.macroActionName[macro_action] == "get lettuce 2" and isinstance(agent.holding, Plate) and not self.lettuce[1].chopped) or (self.macroActionName[macro_action] == "get badlettuce" and isinstance(agent.holding, Plate) and not self.badlettuce[0].chopped):



                            target_x, target_y = self._findPOitem(agent, 6)

                            knife_item_here = self._findItem(target_x, target_y, "knife")
                            if not (knife_item_here.holding and (isinstance(knife_item_here.holding, Lettuce) or isinstance(knife_item_here.holding, BadLettuce)) and knife_item_here.holding.chopped):
                                # self.macroAgent[idx].cur_macro_action_done = True

                                # 如果拿着盘子去不合理的切菜板了，就要把盘子放下

                                # 1. 收集所有合法 counter 的位置
                                counter_positions = []
                                distance_to_knife = []

                                for x_i in range(self.xlen):
                                    for y_i in range(self.ylen):
                                        if ITEMNAME[agent.pomap[x_i][y_i]] == "counter":
                                            counter_positions.append((x_i, y_i))
                                            distance_to_knife.append(self._calDistance(x_i, y_i, target_x, target_y))

                                # 2. 使用 find_nearest_reachable_target 找到最近的合法 counter
                                agent_x, agent_y = agent.x, agent.y
                                counter_index = self.find_nearest_reachable_target(agent.pomap, agent_x, agent_y, counter_positions)

                                # print(counter_positions[counter_index][0], counter_positions[counter_index][1])

                                primitive_action = self._navigate(agent, counter_positions[counter_index][0], counter_positions[counter_index][1])


                                if self._calDistance(agent.x, agent.y, counter_positions[counter_index][0], counter_positions[counter_index][1]) == 1:
                                    self.macroAgent[idx].cur_macro_action_done = True

                                    
                        """新增3：拿着盘子去取盘子"""
                        if self.macroActionName[macro_action] in ["get plate 1", "get plate 2"] and isinstance(agent.holding, Plate):
                            primitive_action = ACTIONIDX["stay"]







                        if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                            if self.macroActionName[macro_action] in ["get plate 1", "get plate 2", "get dirty plate"] and agent.holding:
                                if isinstance(agent.holding, Food):
                                    if not agent.holding.chopped:
                                        primitive_action = ACTIONIDX["stay"]
                            
                            if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and not agent.holding:
                                primitive_action = ACTIONIDX["stay"]

                            if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion"]:
                                    for knife in self.knife:
                                        if knife.x == target_x and knife.y == target_y:
                                            if isinstance(knife.holding, Food):
                                                if not knife.holding.chopped:
                                                    primitive_action = ACTIONIDX["stay"]
                                                    break                           
                            
                            # 如果目标位置发生了移动
                            if self.macroActionName[macro_action] in ["get tomato", "get lettuce 1", "get lettuce 2", "get badlettuce", "get onion", "get plate 1", "get plate 2", "get dirty plate"]:
                                macroAction2Item = {}
                                if self.macroActionName[macro_action] == "get tomato 1":
                                    macroAction2Item["get tomato 1"] = self.tomato[0]
                                elif self.macroActionName[macro_action] == "get tomato 2":
                                    macroAction2Item["get tomato 2"] = self.tomato[1]
                                elif self.macroActionName[macro_action] == "get lettuce 1":
                                    macroAction2Item["get lettuce 1"] = self.lettuce[0]
                                elif self.macroActionName[macro_action] == "get lettuce 2":
                                    macroAction2Item["get lettuce 2"] = self.lettuce[1]
                                elif self.macroActionName[macro_action] == "get badlettuce":
                                    macroAction2Item["get badlettuce"] = self.badlettuce[0]
                                elif self.macroActionName[macro_action] == "get onion":
                                    macroAction2Item["get onion"] = self.onion[0]
                                elif self.macroActionName[macro_action] == "get plate 1":
                                    macroAction2Item["get plate 1"] = self.plate[0]
                                elif self.macroActionName[macro_action] == "get plate 2":
                                    macroAction2Item["get plate 2"] = self.plate[1]
                                elif self.macroActionName[macro_action] == "get dirty plate":
                                    macroAction2Item["get dirty plate"] = self.dirtyplate[0]

                                item = macroAction2Item[self.macroActionName[macro_action]]
                                if target_x != item.x or target_y != item.y:
                                    primitive_action = ACTIONIDX["stay"]

            # 返回的其实只是两个agent下一个step要做的primitive action，而不是一个primitive action序列
            primitive_actions.append(primitive_action)
        return primitive_actions, real_execute_macro_actions


    # A star
    def _navigate(self, agent, target_x, target_y):

        """
        Parameters
        ----------
        agent : Agent
            The current agent.
        target_x : int
            X position of the target item.
        target_y : int
            Y position of the target item.                 

        Returns
        -------
        action : int
            The primitive-action for the agent to choose.
        """

        direction = [(0,1), (0,-1), (1,0), (-1,0)]
        actionIdx = [0, 2, 1, 3]

        # make the agent explore up and down first to aviod deadlock when going to the knife
        q = PriorityQueue()
        q.put(AStarAgent(agent.x, agent.y, 0, self._calDistance(agent.x, agent.y, target_x, target_y), None, [], 0))
        isVisited = [[False for col in range(self.ylen)] for row in range(self.xlen)]
        isVisited[agent.x][agent.y] = True

        while not q.empty():
            aStarAgent = q.get()

            for action in range(4):
                new_x = aStarAgent.x + direction[action][0]
                new_y = aStarAgent.y + direction[action][1]
                new_name = ITEMNAME[agent.pomap[new_x][new_y]] 

                if not isVisited[new_x][new_y]:
                    init_action = None
                    if aStarAgent.action is not None:
                        init_action = aStarAgent.action
                    else:
                        init_action = actionIdx[action]

                    if new_name == "space" or new_name == "agent":
                        pass_agent = 0
                        if new_name == "agent":
                            pass_agent = 1
                        g = aStarAgent.g + 1
                        f = g + self._calDistance(new_x, new_y, target_x, target_y)
                        q.put(AStarAgent(new_x, new_y, g, f, init_action, aStarAgent.history_action + [actionIdx[action]], pass_agent))
                        isVisited[new_x][new_y] = True
                    if new_x == target_x and new_y == target_y:
                        return init_action
        #if no path found, stay
        return ACTIONIDX["stay"]

    def _calDistance(self, x, y, target_x, target_y):
        return abs(target_x - x) + abs(target_y - y)
    
    def _calItemDistance(self, agent, item):
        return abs(item.x - agent.x) + abs(item.y - agent.y)

    def _collectCurMacroActions(self):
        # loop each agent
        cur_mac = []
        for agent in self.macroAgent:
            cur_mac.append(agent.cur_macro_action)
        return cur_mac

    def _computeMacroActionDone(self):
        # loop each agent
        mac_done = []
        for agent in self.macroAgent:
            mac_done.append(agent.cur_macro_action_done)
        return mac_done

    def _get_macro_obs(self):

        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """
        if self.mode == "vector":
            # return self._get_macro_vector_obs()
            return self._get_macro_vector_obs_new()
            # return self._get_macro_vector_obs_new2()
        elif self.mode == "image":
            return self._get_macro_image_obs()
          

    def _get_macro_vector_obs(self):

        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """

        macro_obs = []
        for idx, agent in enumerate(self.agent):
            
            """注释掉这个判断，使得obs能够实时更新，无论MA action是否执行完"""
            # if self.macroAgent[idx].cur_macro_action_done:

            obs = []
            for item in self.itemList:
                x = 0
                y = 0
                if (item.x >= agent.x - self.obs_radius and item.x <= agent.x + self.obs_radius and item.y >= agent.y - self.obs_radius and item.y <= agent.y + self.obs_radius) \
                    or self.obs_radius == 0:
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.append(x)
                    obs.append(y)
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)

                    # 切菜板是否装着东西，盘子里是否装着东西
                    if isinstance(item, Plate):
                        if item.containing:
                            obs.append(1)
                        else:
                            obs.append(0)

                    if isinstance(item, DirtyPlate):
                        if item.containing:
                            obs.append(1)
                        else:
                            obs.append(0)

                    if isinstance(item, Knife):
                        if item.holding:
                            obs.append(1)
                        else:
                            obs.append(0)                

                    if isinstance(item, Agent):
                        if item.holding:
                            obs.append(1)
                        else:
                            obs.append(0)


                else:
                    print('进到不可观测区域了')
                    obs.append(0)
                    obs.append(0)
                    if isinstance(item, Food):
                        obs.append(0)

                    # 切菜板是否装着东西，盘子里是否装着东西
                    if isinstance(item, Plate):
                        if item.containing:
                            obs.append(1)
                        else:
                            obs.append(0)

                    if isinstance(item, DirtyPlate):
                        if item.containing:
                            obs.append(1)
                        else:
                            obs.append(0)

                    if isinstance(item, Knife):
                        if item.holding:
                            obs.append(1)
                        else:
                            obs.append(0)                

                    if isinstance(item, Agent):
                        if item.holding:
                            obs.append(1)
                        else:
                            obs.append(0)


            obs += self.oneHotTask
            
            self.macroAgent[idx].cur_macro_obs = obs 
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))
        return macro_obs


    """这个是能够训练出来的obs，而且十分精简，一定要保留"""
    def _get_macro_vector_obs_new(self):
        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """
        # print(self.itemList)
        macro_obs = []

        for idx, agent in enumerate(self.agent):
            obs = []

            # === Part 1: Encode own agent ===
            obs.append(agent.x / self.xlen)
            obs.append(agent.y / self.ylen)

            # identity one-hot: self = [1, 0]
            obs.append(1)
            obs.append(0)

            # holding (is holding flag)
            obs.append(1 if agent.holding else 0)


            # === Part 2: Encode teammate agent ===
            for teammate in self.agent:
                if teammate == agent:
                    continue  # skip self
                obs.append(teammate.x / self.xlen)
                obs.append(teammate.y / self.ylen)

                # identity one-hot: teammate = [0, 1]
                obs.append(0)
                obs.append(1)

                # # teammate holding: only encode whether holding
                # obs.append(1 if teammate.holding else 0)

                # # teammate holding_idx one-hot
                # obs += get_one_hot_index(teammate.holding)

            # === Part 3: Encode items relative to own agent ===
            for item in self.itemList:
                if isinstance(item, Agent):
                    continue  # Agents already encoded separately

                dx = item.x - agent.x
                dy = item.y - agent.y
                rel_x = dx / self.xlen
                rel_y = dy / self.ylen
                obs.append(rel_x)
                obs.append(rel_y)

                # Food chopped progress
                if isinstance(item, Food):
                    obs.append(item.cur_chopped_times / item.required_chopped_times)

                # Plate containing
                if isinstance(item, Plate):
                    obs.append(1 if item.containing else 0)

                # DirtyPlate containing
                if isinstance(item, DirtyPlate):
                    obs.append(1 if item.containing else 0)

                # Knife holding
                if isinstance(item, Knife):
                    obs.append(1 if item.holding else 0)

            # obs.append(self.macroAgent[idx].cur_macro_action_done)
            # obs.append(self.macroAgent[idx].cur_macro_action)
            
            # Save obs for this agent
            self.macroAgent[idx].cur_macro_obs = obs
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))

        return macro_obs



    def _get_macro_vector_obs_new2(self):
        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """

        def _norm_held_item_index(holder):
            """
            返回 holder 所持物品在 itemList 的归一化索引：
            0.0 表示未持有或找不到，(idx+1)/N 表示第 idx 个（1..N）。
            """
            idx = -1
            N = len(self.itemList)

            # 1) 若有显式 holding_item 对象，优先用它
            if hasattr(holder, "holding_item") and holder.holding_item is not None:
                try:
                    idx = self.itemList.index(holder.holding_item)
                except ValueError:
                    idx = -1

            # 2) 某些实现里 agent.holding 直接是对象（而不仅是布尔）
            elif hasattr(holder, "holding") and holder.holding not in (None, False, 0, 0.0, ""):
                # 避免 bool 被当对象
                if not isinstance(holder.holding, (bool, int, float)):
                    try:
                        idx = self.itemList.index(holder.holding)
                    except ValueError:
                        idx = -1
                else:
                    idx = -1

            # 3) 回退：在 itemList 里找 holder 标记（如 item.holder == agent）
            if idx < 0:
                for i, it in enumerate(self.itemList):
                    if getattr(it, "holder", None) is holder:
                        idx = i
                        break

            if N == 0 or idx < 0:
                return 0.0
            return (idx + 1) / N

        macro_obs = []

        for idx, agent in enumerate(self.agent):
            obs = []

            # === Part 1: Encode own agent ===
            obs.append(agent.x / self.xlen)
            obs.append(agent.y / self.ylen)

            # identity one-hot: self = [1, 0]
            obs.append(1)
            obs.append(0)

            # holding flag + 持有物品在 itemList 的归一化索引
            obs.append(1 if agent.holding else 0)
            obs.append(_norm_held_item_index(agent))

            # === Part 2: Encode teammate agent ===
            for teammate in self.agent:
                if teammate == agent:
                    continue  # skip self
                obs.append(teammate.x / self.xlen)
                obs.append(teammate.y / self.ylen)

                # identity one-hot: teammate = [0, 1]
                obs.append(0)
                obs.append(1)

                # teammate holding flag + 归一化索引
                obs.append(1 if teammate.holding else 0)
                obs.append(_norm_held_item_index(teammate))

            # === Part 3: Encode items relative to own agent ===
            for item in self.itemList:
                if isinstance(item, Agent):
                    continue  # Agents already encoded separately

                dx = item.x - agent.x
                dy = item.y - agent.y
                rel_x = dx / self.xlen
                rel_y = dy / self.ylen
                obs.append(rel_x)
                obs.append(rel_y)

                # Food chopped progress
                if isinstance(item, Food):
                    obs.append(item.cur_chopped_times / item.required_chopped_times)

                # Plate containing
                if isinstance(item, Plate):
                    obs.append(1 if item.containing else 0)

                # DirtyPlate containing
                if isinstance(item, DirtyPlate):
                    obs.append(1 if item.containing else 0)

                # Knife holding
                if isinstance(item, Knife):
                    obs.append(1 if item.holding else 0)

            # 保存
            self.macroAgent[idx].cur_macro_obs = obs
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs, dtype=np.float32))

        return macro_obs
    


    """这个是以partner为主的trajectory"""
    def _get_macro_vector_obs_for_ABImodel(self):
        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """
        # print(self.itemList)
        macro_obs = []

        for idx, agent in enumerate(self.agent):
            obs = []

            # === Part 1: Encode own agent ===
            obs.append(agent.x / self.xlen)
            obs.append(agent.y / self.ylen)

            # identity one-hot: self = [1, 0]
            # obs.append(1)
            # obs.append(0)

            # holding (is holding flag)
            # obs.append(1 if agent.holding else 0)

            # obs.append(self.itemList[])


            # === Part 2: Encode teammate agent ===
            # for teammate in self.agent:
            #     if teammate == agent:
            #         continue  # skip self
            #     obs.append(teammate.x / self.xlen)
            #     obs.append(teammate.y / self.ylen)

            #     # identity one-hot: teammate = [0, 1]
            #     obs.append(0)
            #     obs.append(1)

            #     # # teammate holding: only encode whether holding
            #     # obs.append(1 if teammate.holding else 0)

            #     # # teammate holding_idx one-hot
            #     # obs += get_one_hot_index(teammate.holding)

        # === New Part: Encode held item's index and state ===
            if agent.holding:

                if isinstance(agent.holding, Lettuce):
                    obs.append(1)
                    obs.append(0)
                    obs.append(0)

                if isinstance(agent.holding, BadLettuce):
                    obs.append(0)
                    obs.append(1)
                    obs.append(0)

                if isinstance(agent.holding, Plate):
                    obs.append(0)
                    obs.append(0)
                    obs.append(1)

                # 2. 加入手持物品的状态（根据类型）
                item = agent.holding
                if isinstance(item, Food):
                    obs.append(item.cur_chopped_times / item.required_chopped_times)
                elif isinstance(item, Plate):
                    obs.append(1 if item.containing else 0)
                elif isinstance(item, DirtyPlate):
                    obs.append(1 if item.containing else 0)
                elif isinstance(item, Knife):
                    obs.append(1 if item.holding else 0)
                else:
                    obs.append(0)  # 默认状态为 0
            else:
                # 没有手持物品时补零
                obs.append(0)  # index
                obs.append(0)
                obs.append(0)
                obs.append(0)  # status



            # # === Part 3: Encode items relative to own agent ===
            # for item in self.itemList:
            #     if isinstance(item, Agent):
            #         continue  # Agents already encoded separately

            #     dx = item.x - agent.x
            #     dy = item.y - agent.y
            #     rel_x = dx / self.xlen
            #     rel_y = dy / self.ylen
            #     obs.append(rel_x)
            #     obs.append(rel_y)

            #     # Food chopped progress
            #     if isinstance(item, Food):
            #         obs.append(item.cur_chopped_times / item.required_chopped_times)

            #     # Plate containing
            #     if isinstance(item, Plate):
            #         obs.append(1 if item.containing else 0)

            #     # DirtyPlate containing
            #     if isinstance(item, DirtyPlate):
            #         obs.append(1 if item.containing else 0)

            #     # Knife holding
            #     if isinstance(item, Knife):
            #         obs.append(1 if item.holding else 0)

            # obs.append(self.macroAgent[idx].cur_macro_action_done)
            # obs.append(self.macroAgent[idx].cur_macro_action)
            
            # Save obs for this agent
            self.macroAgent[idx].cur_macro_obs = obs
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))

        return macro_obs




    """这个是以partner为主的trajectory"""
    def _get_macro_vector_obs_for_ABImodel_new(self):
        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """
        # print(self.itemList)
        macro_obs = []

        for idx, agent in enumerate(self.agent):
            obs = []

            # === Part 1: Encode own agent ===
            obs.append(agent.x / self.xlen)
            obs.append(agent.y / self.ylen)

            # identity one-hot: self = [1, 0]
            # obs.append(1)
            # obs.append(0)

            # holding (is holding flag)
            obs.append(1 if agent.holding else 0)

            # obs.append(self.itemList[])


            # === Part 2: Encode teammate agent ===
            # for teammate in self.agent:
            #     if teammate == agent:
            #         continue  # skip self
            #     obs.append(teammate.x / self.xlen)
            #     obs.append(teammate.y / self.ylen)

            #     # identity one-hot: teammate = [0, 1]
            #     obs.append(0)
            #     obs.append(1)

            #     # # teammate holding: only encode whether holding
            #     # obs.append(1 if teammate.holding else 0)

            #     # # teammate holding_idx one-hot
            #     # obs += get_one_hot_index(teammate.holding)

        # === New Part: Encode held item's index and state ===
            if agent.holding:
                # 1. 找出手持物品在 itemList 中的 index
                holding_index = -1
                for i, item in enumerate(self.itemList):
                    if item == agent.holding:
                        holding_index = i
                        break
                obs.append(holding_index / len(self.itemList))  # 归一化 index

                # 2. 加入手持物品的状态（根据类型）
                item = agent.holding
                if isinstance(item, Food):
                    obs.append(item.cur_chopped_times / item.required_chopped_times)
                elif isinstance(item, Plate):
                    obs.append(1 if item.containing else 0)
                elif isinstance(item, DirtyPlate):
                    obs.append(1 if item.containing else 0)
                elif isinstance(item, Knife):
                    obs.append(1 if item.holding else 0)
                else:
                    obs.append(0)  # 默认状态为 0
            else:
                # 没有手持物品时补零
                obs.append(0)  # index
                obs.append(0)  # status



            # === Part 3: Encode items relative to own agent ===
            for item in self.itemList:
                if isinstance(item, Agent):
                    continue  # Agents already encoded separately

                dx = item.x - agent.x
                dy = item.y - agent.y
                rel_x = dx / self.xlen
                rel_y = dy / self.ylen
                obs.append(rel_x)
                obs.append(rel_y)

                # Food chopped progress
                if isinstance(item, Food):
                    obs.append(item.cur_chopped_times / item.required_chopped_times)

                # Plate containing
                if isinstance(item, Plate):
                    obs.append(1 if item.containing else 0)

                # DirtyPlate containing
                if isinstance(item, DirtyPlate):
                    obs.append(1 if item.containing else 0)

                # Knife holding
                if isinstance(item, Knife):
                    obs.append(1 if item.holding else 0)

            # obs.append(self.macroAgent[idx].cur_macro_action_done)
            # obs.append(self.macroAgent[idx].cur_macro_action)
            
            # Save obs for this agent
            self.macroAgent[idx].cur_macro_obs = obs
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))

        return macro_obs
    

    def _get_macro_image_obs(self):

        """
        Returns
        -------
        macro_image_obs : list
            image observation for each agent.
        """
        
        macro_obs = []
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                frame = self.game.get_image_obs()
                if self.obs_radius > 0:
                    old_image_width, old_image_height, channels = frame.shape

                    new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
                    new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
                    color = (0,0,0)
                    obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

                    x_center = (new_image_width - old_image_width) // 2
                    y_center = (new_image_height - old_image_height) // 2

                    obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame
                    obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)

                    self.macroAgent[idx].cur_macro_obs = obs 
                else:
                    self.macroAgent[idx].cur_macro_obs = frame 
            macro_obs.append(self.macroAgent[idx].cur_macro_obs)
        return macro_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n
    


    def intelligently_find_item_number(self, agent_item, raw_name):
        if raw_name == "get lettuce" or raw_name == "get plate":
            macroActionDict = {"stay": 0, "get lettuce 1": 1, "get lettuce 2": 2, "get badlettuce": 3, "get plate 1": 4, "get plate 2": 5, "go to knife 1": 6, "go to knife 2": 7, "deliver 1": 8, "deliver 2": 9, "chop": 10, "go to counter": 11, "take plate to knife": 12, "right": 13, "down": 14, "left": 15, "up": 16}
            target_x_1, target_y_1 = self._findPOitem(agent_item, macroActionDict[raw_name + " 1"])


            can_reach_1 = self._navigate(agent_item, target_x_1, target_y_1)
            distance_1 = self._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

            target_x_2, target_y_2 = self._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
            can_reach_2 = self._navigate(agent_item, target_x_2, target_y_2)
            distance_2 = self._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

            best_action = "stay"
            if can_reach_1 == 4 and can_reach_2 != 4:
                best_action = raw_name + " 2"

            if can_reach_1 != 4 and can_reach_2 == 4:
                best_action = raw_name + " 1"

            if can_reach_1 != 4 and can_reach_2 != 4:
                if distance_1 <= distance_2:
                    best_action = raw_name + " 1"

                else:
                    best_action = raw_name + " 2"
            
            return best_action
        else:
            return raw_name
        








    def shortest_path_through_zeros(self, matrix, start_x, start_y, end_x, end_y):
        """
        起点和终点可以不是0，但中间只能经过0。
        若无法从起点到终点，返回 -1。
        """
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        queue = deque()

        queue.append((start_x, start_y, 0))
        visited[start_x][start_y] = True

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while queue:
            x, y, steps = queue.popleft()

            if x == end_x and y == end_y:
                return steps

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < rows and 0 <= ny < cols and
                    not visited[nx][ny]
                ):
                    # 允许起点或终点不是 0，但中间必须是 0
                    if (nx == end_x and ny == end_y) or matrix[nx][ny] == 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny, steps + 1))

        return -1  # 无法到达
    



    def find_nearest_reachable_target(self, matrix, start_x, start_y, end_points):
        """
        从起点出发（可以不是0），只能经过0，寻找可达终点中最短路径的那个，返回其在 end_points 中的索引。
        若均不可达，返回 -1。
        """
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        queue = deque()
        
        # 将所有终点放入集合，方便 O(1) 查找
        end_set = set(end_points)
        
        queue.append((start_x, start_y, 0))
        visited[start_x][start_y] = True

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while queue:
            x, y, steps = queue.popleft()

            if (x, y) in end_set:
                return end_points.index((x, y))  # 返回原始列表中的索引

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < rows and 0 <= ny < cols and
                    not visited[nx][ny]
                ):
                    if (nx, ny) in end_set or matrix[nx][ny] == 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny, steps + 1))

        return -1  # 全部终点都不可达
    

    def boltzmann_action_from_policy(model, obs, beta: float = 1.0, print_probs: bool = True):
        """
        适用于 SB3 的策略类算法（PPO/A2C 等）且动作为 Discrete。
        beta 越大越“理性”，beta→0 越随机。
        """
        obs_t, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)     # SB3 distribution
            logits = dist.distribution.logits               # [B, n_actions]
            scaled_logits = logits * beta                   # 控制理性程度

            # 转成概率分布
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]

            # 打印（按概率降序）
            # if print_probs:
            #     action_probs = list(zip(macroActionName, probs))
            #     action_probs.sort(key=lambda x: x[1], reverse=True)
            #     print("Boltzmann Probabilities (beta={}):".format(beta))
            #     for name, p in action_probs:
            #         print(f"{name:<20s} {p:.4f}")

            # 按分布采样动作
            m = torch.distributions.Categorical(logits=scaled_logits)
            action = m.sample()
        return int(action.cpu().numpy()[0])
