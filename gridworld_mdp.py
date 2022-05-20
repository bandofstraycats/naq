import numpy as np

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

UP_RIGHT = 4
UP_LEFT = 5
DOWN_RIGHT = 6
DOWN_LEFT = 7

actions_list = [UP, DOWN, RIGHT, LEFT] #, UP_RIGHT, UP_LEFT, DOWN_RIGHT, DOWN_LEFT]

class GridworldMDP():

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def __init__(self, plan_file='plan0.txt', gamma=0.9, random_slide=0.0, act_break_ties=False):
        grid_map = self._read_grid_map(plan_file)
        shape = grid_map.shape

        nS = shape[0] * shape[1]
        nA = len(actions_list)

        MAX_X = shape[0]
        MAX_Y = shape[1]

        grid_values = np.arange(0, nS).reshape(shape)

        def get_xy(ss):
            x, y = np.where(grid_values == ss)
            return x,y
        def ns_up(ss):
            x,y = get_xy(ss)
            return ss if x == 0 or grid_map[x - 1, y] == 1 else ss - MAX_Y
        def ns_down(ss):
            x,y = get_xy(ss)
            return ss if x == MAX_X - 1 or grid_map[x + 1, y] == 1 else ss + MAX_Y
        def ns_right(ss):
            x,y = get_xy(ss)
            return ss if y == MAX_Y - 1 or grid_map[x, y + 1] == 1 else ss + 1
        def ns_left(ss):
            x,y = get_xy(ss)
            return ss if y == 0 or grid_map[x, y - 1] == 1 else ss - 1

        ns_up_right = lambda ss: ns_right(ns_up(ss))
        ns_up_left = lambda ss: ns_left(ns_up(ss))
        ns_down_right = lambda ss: ns_right(ns_down(ss))
        ns_down_left = lambda ss: ns_left(ns_down(ss))

        actions_func_list = [ns_up, ns_down, ns_right, ns_left, ns_up_right, ns_up_left, ns_down_right,
                             ns_down_left]

        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))

        it = np.nditer(grid_map, flags=['multi_index'])
        state_vars = np.zeros((nS, 3))

        while not it.finished:
            s = it.iterindex
            x, y = it.multi_index

            pos = np.array([x, y])
            #target_dist = np.linalg.norm(target - pos)
            #target_theta = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            state_vars[s] = [(s+1.)/(MAX_X*MAX_Y), (x+1.)/MAX_X, (y+1.)/MAX_Y]
            #state_vars[s] = [target_dist, np.cos(target_theta), np.sin(target_theta)]

            is_done = grid_map[x, y] == 3 or grid_map[x, y] == 1

            reward = -1.0
            if grid_map[x, y] == 3:
                reward = 0.0
            elif grid_map[x, y] == 1:
                reward = 0.0
            elif grid_map[x, y] == 2:
                reward = -5.0

            bonus = [-0.01, 0.01, -0.05, 0.05]
            for a in range(nA):
                R[s, a] = reward + bonus[a] if act_break_ties else reward

            # Terminal state
            if is_done:
                for a in range(nA):
                    P[s, a, s] = 1.0
            # Not a terminal state
            else:
                for action, func in zip(actions_list, actions_func_list[:len(actions_list)]):
                    if random_slide > 0:
                        P[s, action, func(s)] = 1.0 - 2 * random_slide
                        P[s, action, ns_up(func(s))] += random_slide
                        P[s, action, ns_down(func(s))] += random_slide
                        #P[s, action, ns_left(func(s))] += random_slide
                    else:
                        P[s, action, func(s)] = 1.0

            it.iternext()

        # Initial state distribution is uniform
        self.d_0 = np.reshape(np.ones(nS) / nS, [-1, 1])

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.R = R

        # gamma
        self.gamma = gamma
        # counts
        self.nS = nS
        self.nA = nA
        self.shape = shape
        # state features
        self.state_vars = state_vars
        #print(state_vars)

        # test
        self.test_p_stochastic()

    def mdp_to_lp_primal(self):
        """
        min d_0^T v
        s.t. Ax >= b
        :return: A,b
        """
        A_a = list()
        for a in range(self.nA):
            A_a.append(np.identity(self.nS) - self.gamma * self.P[:, a, :])

        A = np.concatenate(A_a, axis=0)
        assert np.shape(A) == (self.nS * self.nA, self.nS)

        R_rav = np.reshape(np.transpose(self.R), [-1])
        assert np.shape(R_rav) == (self.nS * self.nA, )

        return A, R_rav

    def test_p_stochastic(self):
        for s in range(self.nS):
            np.testing.assert_array_almost_equal(self.P[s, :, :].sum(axis=1), np.ones(self.nA), decimal=2)
            # print(self.P[s, :, :])
            # print('----')
        print("P is stochastic")
