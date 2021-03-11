from scipy.stats import binom
import numpy as np
import itertools
import time

class zanon_model:
    
    """
    A class used to define the z-anonymity model
    
    ...
    
    Attributes
    ----------
    z : int
        the z threshold relative to the z-anonymity algorithm
    dt : int
        the eviction time
    U : int
        number of users in the system
    A : int
        attribute catalog size
    k : int
        k-anonymity target
    theta1 : float
        attribute's popularity threshold
    theta2 : float
        realization's popularity threshold
    lambda0 : float
        the most popular attribute's exposing rate
    l : list
        attributes' exposing rate list
    px : list
        attributes' exposing probability list
    po : list
        attributes' publishing probability list, if the attribute
        has been exposed
    py : list
        attributes' publishing probability list
    pkanon : float
        probability of a user in the system being k-anonymized
    Aeff : int
        the number of effective attributes
    pseq_star : float
        effective realization probability threshold
    pseqs : list
        list of effective realization probability
    flipping_order : list
        attributes' indexes ordered by the probability of flipping
        their value with respect to their most likely one
    
    
    Methods
    -------
    get_lambda0()
        Returns the value of lambda0 for this instance of the model
    get_l()
        Returns the value of l for this instance of the model
    get_px()
        Returns the value of px for this instance of the model
    get_po()
        Returns the value of po for this instance of the model
    get_py()
        Returns the value of py for this instance of the model
    get_Aeff()
        Returns the value of Aeff for this instance of the model
    get_pseq_star()
        Returns the value of pseq_star for this instance of the model
    get_pseqs()
        Returns the value of pseqs for this instance of the model
    get_pkanon()
        Returns the value of pkanon for this instance of the model
    get_flipping_order()
        Returns the value of flipping_order for this instance of the model
    """
    
    po = []
    py = []

    pkanon = -1
    Aeff = -1
    pseq_star = -1
    pseqs = []
    flipping_order = []
    
    def __init__(self, z, dt, U, A, k, theta1, theta2, lambda0 = 0., l = [], px = []):
        """
        Parameters
        ----------
        z : int
            the z threshold relative to the z-anonymity algorithm
        dt : int
            the eviction time
        U : int
            number of users in the system
        A : int
            attribute catalog size
        k : int
            k-anonymity target
        theta1 : float
            attribute's popularity threshold
        theta2 : float
            realization's popularity threshold
        lambda0 : float
            the most popular attribute's exposing rate
        l : list, optional
            attributes' exposing rate list (default is an empty list)
        px : list, optional
            attributes' exposing probability list (default is an empty list)
        """
        
        self.z = z
        self.dt = dt
        self.U = U
        self.A = A
        self.lambda0 = lambda0
        self.k = k
        self.l = np.array(l)
        self.px = np.array(px)
        self.theta1 = theta1
        self.theta2 = theta2
    
    def get_lambda0(self):
        """Returns the input lambda0 or, if invalid, raises an error
        
        Returns
        -------
        float
            the lambda0 input by the user
        
        Raises
        ------
        ValueError
            if the provided lambda0 is invalid
        """
        if self.lambda0 > 0:
            return self.lambda0
        else:
            raise ValueError('Insert at least one among lambda0, l, and px, or a valid lambda0')
    
    def get_l(self):
        """Returns the exposing rate list for this instance of the model.
        
        If `l` has not been provided at instantiation time,
        it creates a default `l` with a 1/x trend, with
        initial value `dt` * `lambda0`.
        
        Returns
        -------
        list
            The exposing rate list, one per attribute
        """
        
        if len(self.l) == 0:
            self.l = np.array([self.dt * self.get_lambda0() / x for x in range(1, self.A + 1)])
        return self.l
    
    def get_px(self):
        """Returns the exposing probability list in a Delta t for this instance of the model.
        
        If `px` has not been provided at instantiation time,
        it creates a default `px` value using the `l` attribute
        
        Returns
        -------
        list
            The exposing probability list, one per attribute
        """
        
        if len(self.px) == 0:
            self.px = 1 - np.power(np.e, - self.get_l())
        return self.px
        
    def get_po(self):
        """Returns the publishing probability list in a Delta t for this instance of the model,
        assuming that attribute a is exposed
        
        Returns
        -------
        list
            The publishing probability list assuming that attribute is exposed,
            one per attribute
        """
        
        if len(self.po) == 0:
            self.po = 1 - binom.cdf(self.z - 2, self.U - 1, self.get_px())
        return self.po
    
    def get_py(self):
        """Returns the publishing probability list in a Delta t for this instance of the model
        
        Returns
        -------
        list
            The publishing probability list, one per attribute
        """
        
        if len(self.py) == 0:
            self.py = self.get_px() * self.get_po()
        return self.py
    
    def get_Aeff(self):
        """Returns the number of effective attributes for this instance of the model
        
        Returns
        -------
        int
            The number of effective attributes
        """
        
        if self.Aeff == -1:
            self.Aeff = np.sum(self.get_py() >= self.theta1 / self.U)
        return self.Aeff 
    
    def get_pseq_star(self):
        """Returns the effective realization threshold for this instance of the model
        
        Returns
        -------
        float
            The effective realization threshold
        """
        
        if self.pseq_star == -1:
            self.pseq_star = self.theta2 / (2 ** self.get_Aeff())
        return self.pseq_star
    
    def get_pseqs(self):
        """Returns the effective realizations' probabilities for this instance of the model
        
        Returns
        -------
        list
            List of effective realizations' probabilities
        """
        
        if len(self.pseqs) == 0:
            #instantiate the most likely realization for this py
            seq0 = ['1' if attr >.5 else '0' for attr in self.get_py()[:self.get_Aeff()]]
            seq0 = int('0b' + ''.join(seq0) if len(seq0) != 0 else '0b0', 2)
            self.__explore_branch(seq0, 0)
        return self.pseqs
    
    def get_pkanon(self):
        """Returns the value of a user being k-anonymized for this instance of the model
        
        Returns
        -------
        float
            The proability of a user being k-anonymized
        """
        
        if self.pkanon == -1:
            pseqs = self.get_pseqs()
            self.pkanon = 0
            for pseq in pseqs:
                self.pkanon += (1 - binom.cdf(self.k - 2, self.U - 1, pseq)) * pseq
        return self.pkanon
    
    def get_flipping_order(self):
        """Returns the value of flipping_order for this instance of the model
        
        Returns
        -------
        list
            List of attributes, ordered by their probability of assuming the opposite value
            with respect to the most likely one
        """
        if len(self.flipping_order) == 0:
            distance_from_half = np.abs(self.get_py()[:self.get_Aeff()] - .5)
            self.flipping_order = np.argsort(distance_from_half)
            self.flipping_order = len(self.flipping_order) - self.flipping_order - 1
        return self.flipping_order
    
    def __explore_branch(self, node, loc_start):
        """Recursive method to explore the realization tree
        
        Parameters
        ----------
        node : int
            The parent node
        loc_start : The first attribute to evaluate
        
        Returns
        -------
        boolean
            Whether a branch has to be explored with more detail or not"""
        
        dead_branch = False
        #evaluate node p_seq
        pseq = self.__evaluate_pseq(node)
        if pseq > self.get_pseq_star():
            #add p_seq if above threshold
            self.pseqs.append(pseq)
            for loc in range(loc_start, self.get_Aeff()): #check for every child
                #evaluate the new node
                child = self.__evaluate_child(node, loc)
                #recur
                dead_child = self.__explore_branch(child, loc + 1)
                #if a child node pseq is below threshold stop the children exploration
                if dead_child:
                    break
        else: #if the pseq is below threshold
            dead_branch = True
        
        return dead_branch

    def __evaluate_child(self, parent, loc):
        """Return the child node's bitstring under the form of an integer
        
        Every bitstring of known length can be described by one and one only integer.
        E.g.:
            '000' -> 0
            '010' -> 2
            '110' -> 6
        
        Parameters
        ----------
        parent : int
            The parent node in the form of an integer
        loc: int
            The attribute that has to be flipped, according to the ordering set in
            `flipping_order`
        
        Return
        ------
        int
            bitstring under the form of an integer
        """
        
        child = parent^(2 ** self.get_flipping_order()[loc])
        return child

    def __evaluate_pseq(self, node):
        """Returns the probability of a realization to happen
        
        Parameters
        ----------
        node : int
            The tree node to evaluate
            
        Returns
        -------
        float
            The probability of a realization to happen
        """
        
        seq = self.__get_bitstring(node)[2:]
        pseq = np.prod(np.array([int(seq[a]) * self.get_py()[a] + (1 - int(seq[a])) * (1 - self.get_py()[a]) for a in range(self.get_Aeff())]))
        return pseq
    
    def __get_bitstring(self, x):
        """Return the input node in a bitstring form
        
        Parameters
        ----------
        x : int
            The node in an integer form
        
        Returns
        -------
        str
            The node's string form
        """
        
        y = format(x, "#0{}b".format(self.get_Aeff() + 2))
        return y