from scipy.stats import binom
import numpy as np
import itertools
import time

class zanon_model_classes:
    
    """
    A class used to define the z-anonymity model. In the current configuration,
    the class is developed tomanage two classes of users
    
    ...
    
    Attributes
    ----------
    z : int
        the z threshold relative to the z-anonymity algorithm
    dt : int
        the eviction time
    U : list
        number of users in the system, one item of the list per class
    A : int
        attribute catalog size
    k : int
        k-anonymity target
    theta1 : float
        attribute's popularity threshold parameter
    theta2 : float
        realization's popularity threshold parameter
    lambda0 : numpy.ndarray
        the most popular attribute's exposing rate, one item of the array per class
    l : numpy.ndarray
        attributes' exposing rates, with dimension (number of attributes x number of classes)
    px : numpy.ndarray
        attributes' exposing probabilities, with dimension (number of attributes x number of classes)
    po : numpy.ndarray
        1-D array containing attributes' publishing probabilities, if the attribute
        has been exposed. Differing from what it is written in the paper,
        it is not dependent on the class: the Kronecker delta is ignored.
        It has dimensions
    py : numpy.ndarray
        attributes' publishing probability list, with dimension (number of attributes x number of classes)
    pkanon : list
        probability of a user in the system being k-anonymized, one value per class
    Aeff : int
        the number of effective attributes
    pseq_star : float
        effective realization probability threshold
    pseqs : dict
        Python dictionary of effective realization probabilities. The key is the node,
        the value another dictionary with the class as key and the pseq as value
    flipping_order : list of list
        attributes' indexes ordered by the probability of flipping
        their value with respect to their most likely one, one list per class
    
    
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
    """
    
    po = []
    py = []

    pkanon = []
    Aeff = -1
    pseqs = dict()
    pseq_star = -1
    flipping_order = [[],[]]
    
    def __init__(self, z, dt, U, A, k, theta1, theta2, lambda0 = [], l = [], px = []):
        
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
            attribute's popularity threshold parameter
        theta2 : float
            realization's popularity threshold parameter
        lambda0 : list, optional
            the most popular attribute's exposing rate, one item per class (default is an empty list)
        l : list, optional
            attributes' exposing rate list, one list per attribute (default is an empty list)
        px : list, optional
            attributes' exposing probability list, one list per attribute (default is an empty list)
        """
        
        self.z = z
        self.dt = dt
        self.U = U
        self.A = A
        self.k = k
        self.theta1 = theta1
        self.theta2 = theta2
        self.lambda0 = np.array(lambda0)
        self.l = np.array(l)
        self.px = np.array(px)
    
    def get_lambda0(self):
        """Returns the input lambda0 or, if invalid, raises an error
        
        Returns
        -------
        numpy.ndarray
            the lambda0 input by the user
        
        Raises
        ------
        ValueError
            if the provided lambda0 is invalid
        """
        if len(self.lambda0) == 2:
            return self.lambda0
        else:
            raise ValueError('Insert at least one among lambda0, l, and px, or a valid lambda0')
    
    def get_l(self):
        """Returns the exposing rates for this instance of the model.
        
        If `l` has not been provided at instantiation time,
        it creates a default `l` with a 1/x trend, with
        initial value `dt` * `lambda0`. The process is parallelized
        for each class
        
        Returns
        -------
        numpy.ndarray
            The exposing rates, with dimension (number of attributes x number of classes)
        """
        if len(self.l) == 0:
            self.l = np.array([self.dt * self.lambda0 / x for x in range(1, self.A + 1)])
        return self.l
    
    def get_px(self):
        """Returns the exposing probabilities in a Delta t for this instance of the model.
        
        If `px` has not been provided at instantiation time,
        it creates a default `px` value using the `l` attribute
        
        Returns
        -------
        numpy.ndarray
            The exposing probabilities, with dimension (number of attributes x number of classes)
        """
        
        if len(self.px) == 0:
            self.px = 1 - np.power(np.e, - self.get_l())
        return self.px
        
    def get_po(self):
        """Returns the publishing probabilities in a Delta t for this instance of the model,
        assuming that attribute a is exposed
        
        Returns
        -------
        numpy.ndarray
            The publishing probabilities assuming that attribute is exposed,
            one per attribute
        """
        if len(self.po) == 0:
            self.po = np.array([1.] * self.A)
            for i in range(self.z - 1): #number of users of the first class
                for j in range(self.z - 1 - i): #number of users of the second class
                    self.po -= binom.pmf(i, self.U[0], self.get_px()[:,0]) * binom.pmf(j, self.U[1], self.get_px()[:,1])
        return self.po
    
    def get_py(self):
        """Returns the publishing probabilities in a Delta t for this instance of the model
        
        Returns
        -------
        numpy.ndarray
            The publishing probabilities, with dimension (number of attributes x number of classes)
        """
        
        if len(self.py) == 0:
            self.py = (self.get_px().T * self.get_po()).T
        return self.py
    
    def get_Aeff(self):
        """Returns the number of effective attributes for this instance of the model
        
        Returns
        -------
        int
            The number of effective attributes
        """
        if self.Aeff == -1:
            #count how many attributes have on average a better py than the threshold
            self.Aeff = np.sum(np.average(self.get_py(), axis = 1, weights = self.U) >= (self.theta1 / sum(self.U)))
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
        dict of dict
            Python dictionary of effective realizations' probabilities. The key is the realization,
            the value another dictionary with the class as key and the pseq as value
        """
        
        if len(self.pseqs.keys()) == 0:
            #most probable realization for the first class
            seq00 = ['1' if attr >.5 else '0' for attr in self.get_py()[:self.get_Aeff(), 0]]
            seq00 = int('0b' + ''.join(seq00) if len(seq00) != 0 else '0b0', 2)
            self.__explore_branch(seq00, 0, 0) #look for other realizations
            
            #most probable realization for the second class
            seq01 = ['1' if attr >.5 else '0' for attr in self.get_py()[:self.get_Aeff(), 1]]
            seq01 = int('0b' + ''.join(seq01) if len(seq01) != 0 else '0b0', 2)
            self.__explore_branch(seq01, 0, 1) #look for other realizations
            
            #for realizations who were found only in one class
            #add 0-value pseqs on the other class
            for seq in self.pseqs.keys():
                if len(self.pseqs[seq]) == 1:
                    if 0 in self.pseqs[seq]:
                        self.pseqs[seq][1] = 0.
                    else:
                        self.pseqs[seq][0] = 0.
            
        return self.pseqs
    
    def get_pkanon(self):
        
        """Returns the value of a user being k-anonymized for this instance of the model, one value per class
        
        Returns
        -------
        list
            The proabilities of a user being k-anonymized
        """
        
        if len(self.pkanon) == 0:
            pseqs = self.get_pseqs()
            binoms = dict()
            
            #the binomial for each realization is evaluated once for all in advanced
            for seq in pseqs.keys():
                for i in range(self.k - 1):
                    binoms[seq] = dict()
                    binoms[seq][0] = dict()
                    binoms[seq][1] = dict()
                    binoms[seq][0][i] = binom.pmf(i, self.U[0], pseqs[seq][0])
                    binoms[seq][1][i] = binom.pmf(i, self.U[1], pseqs[seq][1])
            
            self.pkanon = [0, 0]
            for seq in pseqs.keys():
                pkanon_seq = [1, 1]
                for i in range(self.k - 1): #users of first class
                    for j in range(self.k - 1 - i): #users of second class
                        pkanon_seq[0] -= binoms[seq][0][i] * binoms[seq][1][j]
                        pkanon_seq[1] -= binoms[seq][0][i] * binoms[seq][1][j]
                self.pkanon[0] += pkanon_seq[0] * pseqs[seq][0]
                self.pkanon[1] += pkanon_seq[1] * pseqs[seq][1]
        return self.pkanon
    
    def __get_flipping_order(self, c):
        """Returns the value of flipping_order for this instance of the model
        
        Parameters
        ----------
        c : int
            The class whose flipping order has to be evaluated
        Returns
        -------
        list
            List of attributes, ordered by their probability of assuming the opposite value
            with respect to the most likely one
        """
        if len(self.flipping_order[c]) == 0:
            distance_from_half = np.abs(self.get_py()[:self.get_Aeff(), c] - .5)
            self.flipping_order[c] = np.argsort(distance_from_half)
            self.flipping_order[c] = len(self.flipping_order[c]) - self.flipping_order[c] - 1
        return self.flipping_order[c]
    
    def __explore_branch(self, node, loc_start, c):
        """Recursive method to explore the realization tree
        
        Parameters
        ----------
        node : int
            The parent node
        loc_start : The first attribute to evaluate
        c : int
            The class taken into consideration
        
        Returns
        -------
        boolean
            Whether a branch has to be explored with more detail or not"""
        
        dead_branch = False
        #evaluate node p_seq
        pseq = self.__evaluate_pseq(node, c)
        if pseq > self.get_pseq_star():
            #add p_seq if above threshold
            if node not in self.pseqs:
                self.pseqs[node] = dict()
            self.pseqs[node][c] = pseq
            for loc in range(loc_start, self.get_Aeff()): #check for every child
                #evaluate the new node
                child = self.__evaluate_child(node, loc, c)
                #recur
                dead_child = self.__explore_branch(child, loc + 1, c)
                #if a child node pseq is below threshold stop the children exploration
                if dead_child:
                    break
        else: #if the pseq is below threshold
            dead_branch = True
        
        return dead_branch

    def __evaluate_child(self, parent, loc, c):
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
        c : int
            The class taken into consideration
        
        Return
        ------
        int
            bitstring under the form of an integer
        """
        child = parent^(2 ** self.__get_flipping_order(c)[loc])
        return child

    def __evaluate_pseq(self, node, c):
        """Returns the probability of a realization to happen
        
        Parameters
        ----------
        node : int
            The tree node to evaluate
        c : int
            The class taken into consideration
            
        Returns
        -------
        float
            The probability of a realization to happen
        """
        seq = self.__get_bitstring(node)[2:]
        #print(print_bin(node))
        pseq = np.prod(np.array([int(seq[a]) * self.get_py()[a, c] + (1 - int(seq[a])) * (1 - self.get_py()[a, c]) for a in range(self.get_Aeff())]))
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