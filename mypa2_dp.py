from mymdp import MDP
import math

class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """        
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """
        for state in self.mdp.states():
            self.pi[state] = {}
            n = len(self.mdp.actions(state))
            if n != 0:
                p = 1/n
                for action in self.mdp.actions(state):
                    self.pi[state][action] = p
                    
    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        q = {}
        for state in self.mdp.states():
            q[state] = {}
            for action in self.mdp.actions(state):
                q_value = 0
                for s_prime, prob in self.mdp.T(state, action):
                    q_value += prob * (self.mdp.R(state, action, s_prime) + self.mdp.gamma * v[s_prime])
                q[state][action] = q_value

        return q

    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pi = {}
        q = self.computeq_fromv(v)
        for state in self.mdp.states():
            actions = q[state]
            max_action = max(actions, key=actions.get)
            pi[state] = {}
            for a in actions:
                if a == max_action:
                    pi[state][a] = 1
                else:
                    pi[state][a] = 0
        return pi

    def check_term(self, v: dict[str,float], next_v: dict[str,float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        for state in v:
            if abs(next_v[state] - v[state]) > self.thresh:
                return True
        return False


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        updated_v = {}
        for state in self.mdp.states():
            sum_of_avg_returns_all_actions = 0
            for action in pi[state]:
                probability_pi = pi[state][action]
                sum_of_avg_returns_action = 0
                for s_prime, trans_prob in self.mdp.T(state, action):
                    sum_of_avg_returns_action += trans_prob * (self.mdp.R(state, action, s_prime) + self.mdp.gamma * self.v.get(s_prime))
                sum_of_avg_returns_all_actions += probability_pi * sum_of_avg_returns_action
            updated_v[state] = sum_of_avg_returns_all_actions
        return updated_v


    def policy_iteration(self) -> dict[str,dict[str,float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        You must use:
         - __iter_policy_eval
         - greedy_policy_improvement        

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pass


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy     

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def value_iteration(self) -> dict[str,dict[str,float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        You must use:
         - greedy_policy_improvement           

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pass
