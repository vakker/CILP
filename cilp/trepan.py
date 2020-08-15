# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:03:24 2017

@author: Kester
"""

import numpy as np
from scipy import stats


class Trepan(object):
    """
    A class that represents the TREPAN tree model. Must be initialised with an
    oracle, and then fitted to a specific dataset to build the tree.

    Variables:
        maxsize: maximum number of internal nodes to grow the tree to (before pruning)
        minsamples: the minimum number of samples to use for determining each m-of-n test
        significance: the significance threshold to use when determining if a node
                      has a different distribution of values for a particular feature
                      from the previous node
        improvement: the percentage improvement on the previous test needed to
                     accept an expanded m-of-n test
        oracle: a fitted model used as an oracle


    Public methods:
        fit(samples): Fits a TREPAN tree using the oracle and the samples provided.
        predict(samples): Uses an already fitted tree to predict a class for
                          each of the samples provided.
        draw_tree(filename): Outputs an already fitted tree as a .dot graph to
                             the provided filenname.
        oracle_predict(samples): Predict a class using the oracle for the samples
                                 provided.
    """
    def __init__(self,
                 oracle,
                 maxsize=10,
                 minsamples=1000,
                 significance=0.05,
                 improvement=1.05,
                 verbose=False,
                 logging=False):
        self.maxsize = maxsize
        self.minsamples = minsamples
        self.significance = significance
        self.improvement = improvement
        self.verbose = verbose
        self.logging = logging
        self.tree = {}
        self.fitted = False
        self.building = False
        self.oracle = oracle

    def fit(self, traindata, trainlabels=[], testdata=[], testlabels=[], featnames=[]):
        """
        Takes a set of training data (as a numpy array), saves it as part of
        the class, and calls the build_tree method to make a TREPAN tree.
        """
        if self.fitted:
            raise Exception('This TREPAN model has already been fitted.')
        self.traindata = traindata
        self.trainlabels = trainlabels
        self.testdata = testdata
        self.testlabels = testlabels
        self.numsamples = traindata.shape[0]
        self.numfeats = traindata.shape[1]
        # Check inputs match up
        if self.logging and len(self.trainlabels) != self.numsamples:
            raise Exception('Number of training examples and labels do not match')
        if self.logging and self.testdata.shape[1] != self.numfeats:
            raise Exception('Test data has incorrect number of features')
        if self.logging and len(self.testlabels) != self.testdata.shape[0]:
            raise Exception('Number of test examples and labels do not match')
        if len(featnames):
            self.featnames = featnames
        else:
            self.featnames = [str(i) for i in range(self.numfeats)]
        if len(self.featnames) != self.numfeats:
            raise Exception('Number of feature names does not match number of features')
        # Make sure tree is empty before we build
        self.tree = {}
        self.__build_tree()
        self.fitted = True

    def __build_tree(self):
        """
        A high-level function to build a TREPAN tree, consisting of creating a
        tree and repeatedly expanding it until certain stopping conditions are
        reached, then pruning branches.
        The tree is stored as a dictionary of nodes, with the node name as the
        key.
        """
        # Record that we're currently building the tree
        self.building = True
        # Initialise list of best node values
        self.fvalues = {}
        # Create initial node with name '1'
        self.tree['1'] = self.__create_initial_node('1')
        if self.verbose:
            print("Initialising tree...")
        # Initialise accuracy and fidelity logs if appropriate
        if self.logging:
            self.trainaccs = []
            self.trainfids = []
            self.testaccs = []
            self.testfids = []
        finished = False
        # Repeat node expansion until stopping conditions reached
        while not finished:
            # Find best node with the max f-value in the dictionary
            bestnode = max(self.fvalues, key=lambda x: self.fvalues[x])
            # Expand it
            wasexpanded = self.__expand_node(bestnode)
            # Log if necessary
            if self.logging and wasexpanded:
                self.trainaccs.append(self.accuracy(self.traindata, self.trainlabels))
                self.trainfids.append(self.fidelity(self.traindata))
                if len(self.testdata) > 0:
                    self.testaccs.append(self.accuracy(self.testdata, self.testlabels))
                    self.testfids.append(self.fidelity(self.testdata))
            # Check there are still nodes to expand
            finished = self.__check_stopping()
        self.__prune_tree()
        self.building = False

    def __check_stopping(self):
        """
        Checks whether tree building should be finished. This is either if there
        are no nodes in the queue to expand, or the number of internal nodes
        in the tree is equal to the maximum size specified.
        """
        internalnodes = sum([not self.tree[node]['isleaf'] for node in self.tree])
        if len(self.fvalues) == 0 or internalnodes >= self.maxsize:
            finished = True
        else:
            finished = False
        return finished

    def __create_initial_node(self, nodename):
        """
        Creates the initial node for the tree; a simple wrapper for dict creation.
        Reach and fidelity are both set to 1. Many other variables are set
        to None and will be defined when this node is expanded as the first
        part of the tree-building process.
        See description of __create_node method for explanation of all the
        variables saved in a node.
        """
        constraints = []
        reach = 1
        fidelity = 1
        self.fvalues[nodename] = reach * (1 - fidelity)
        # All training data reaches this node
        reached = [True for i in range(self.numsamples)]
        isleaf = True
        # Predict (ONLY for this initial node) using raw predictions from oracle -
        # no drawn samples
        predictedclass = stats.mode(self.oracle_predict(self.traindata[reached])).mode[0]
        mntest = None
        daughter0 = None
        daughter1 = None
        parent = None
        sister = None
        node = {
            'constraints': constraints,
            'reach': reach,
            'fidelity': fidelity,
            'reached': reached,
            'isleaf': isleaf,
            'predictedclass': predictedclass,
            'mntest': mntest,
            '0daughter': daughter0,
            '1daughter': daughter1,
            'parent': parent,
            'sister': sister
        }
        return node

    def __create_node(self, nodename, parent, prediction, sister, passed):
        """
        Creates a new node using the passed variables. Includes calculation of
        samples that reach this node, plus predicting local labels for a predicted
        class and the fidelity.

        Values in node dictionary:
            constraints: A list of the m-of-n tests that must be passed/failed
                        to reach this node.
            reach: The proportion of total training data that reaches this node.
            reached: A boolean array showing the exact training data examples
                     that reach this node.
            predictedclass: The predicted class at this node (i.e. the modal
                            value predicted by the oracle for samples at the node)
            fidelity: The proportion of samples at this node for which the
                      node's predicted class matches their class as predicted by
                      the oracle.
            mntest: The m-of-n test for this node. m-of-n tests are stored as a
                    tuple with the format (m, [f1, f2...fn]).
                    fn values are also tuples, with the format
                    (feature, threshold, greater), where feature is the integer
                    index of the feature being split on, threshold is the floating
                    point value being checked against, and greater is a boolean
                    which indicates whether a value must be equal or greater than
                    (True) or less than (False) the threshold to count towards
                    passing the test.
                    Finally, when a test is added to a constraints list, it
                    includes a third value, passed, a boolean that indicates
                    whether the test must be passed or failed to reach the
                    relevant node.
            isleaf: A boolean showing whether a node is a leaf node.
            parent: The name of the node's parent node
            0daughter: The name of the node's daughter node when its m-of-n test
                       is failed.
            1daughter: The name of the node's daughter node when its m-of-n test
                       is failed.
            sister: The name of the node's sister node (i.e. they have the same parent)
        """
        # Create constraints from parent's constraints plus its m-of-n test
        newtest = (self.tree[parent]['mntest'][0], self.tree[parent]['mntest'][1], passed)
        constraints = self.tree[parent]['constraints'] + list([newtest])
        # Find how many samples reach the node (as boolean array)
        reached = [
            self.__passes_mn_tests(self.traindata[i, :], constraints)
            for i in range(self.numsamples)
        ]
        reach = sum(reached) / self.numsamples
        # Get labels for samples that reach the node
        localsamples = self.traindata[reached]
        labels = self.oracle_predict(localsamples)
        # Assign predicted classes based on the m-of-n test that created this node
        predictedclass = prediction
        # Use this to calculate fidelity
        if reach > 0:
            fidelity = sum(labels == predictedclass) / sum(reached)
        else:
            fidelity = 0
        # Get all features already used in this branch
        constrainedfeats = []
        for test in constraints:
            for subtest in test[1]:
                constrainedfeats.append(subtest[0])
        # If labels are not all the same, and we haven't used all features,
        # add this to the list of nodes to expand
        if len(np.unique(labels)) > 1 and len(constrainedfeats) < self.numfeats:
            self.fvalues[nodename] = reach * (1 - fidelity)
        isleaf = True
        mntest = None
        daughter0 = None
        daughter1 = None
        node = {
            'constraints': constraints,
            'reach': reach,
            'fidelity': fidelity,
            'reached': reached,
            'isleaf': isleaf,
            'predictedclass': predictedclass,
            'mntest': mntest,
            '0daughter': daughter0,
            '1daughter': daughter1,
            'parent': parent,
            'sister': sister
        }
        return node

    def __passes_mn_test(self, example, test):
        """
        Takes a particular example of test data, and checks whether it passes a
        single m-of-n test that is also provided.
        """
        testpassed = False
        counter = 0
        featurespassed = 0
        # Pull out m and n for clarity
        m = test[0]
        n = len(test[1])
        # Loop until either the test is passed or we get to n features tested
        while (not testpassed) and counter < n:
            # Pull out details for particular subtest
            feature = test[1][counter][0]
            threshold = test[1][counter][1]
            greater = test[1][counter][2]
            # Check if subtest passed
            if (greater and example[feature] >= threshold) or \
               ((not greater) and example[feature] < threshold):
                featurespassed += 1
            # Check if overall test passed
            if featurespassed >= m:
                testpassed = True
            counter += 1
        return testpassed

    def __passes_mn_tests(self, example, constraints):
        """
        Takes an example and  a list of m-of-n tests and checks if all tests
        are passed by that particular sample.
        """
        allpassed = True
        counter = 0
        # Loop over tests until one is failed or we have tested them all
        while allpassed and counter < len(constraints):
            passed = self.__passes_mn_test(example,
                                           (constraints[counter][0], constraints[counter][1]))
            if passed != constraints[counter][2]:
                allpassed = False
            counter += 1
        return allpassed

    def __expand_node(self, nodename):
        """
        Expands a provided node by constructing an m-of-n test and using it to
        create two daughter nodes depending on whether it was passed or failed.
        """
        # Construct m-of-n test
        constructed = self.__construct_test(nodename)
        mntest = constructed[0]
        passclass = constructed[1]
        failclass = constructed[2]
        wasexpanded = False
        # Check we made a test before editing the node - otherwise, skip to the
        # end and just keep this node as an unexpanded leaf
        if mntest is not None:
            # Add test to the node
            self.tree[nodename]['mntest'] = mntest
            # Generate daughter nodes
            daughter0 = nodename + '0'
            daughter1 = nodename + '1'
            if self.verbose:
                print("Creating new nodes...")
            self.tree[daughter0] = self.__create_node(daughter0, nodename, failclass, daughter1,
                                                      False)
            self.tree[daughter1] = self.__create_node(daughter1, nodename, passclass, daughter0,
                                                      True)
            # Adjust the current node's values to register expansion
            self.tree[nodename]['0daughter'] = daughter0
            self.tree[nodename]['1daughter'] = daughter1
            self.tree[nodename]['isleaf'] = False
            wasexpanded = True
        del self.fvalues[nodename]
        # Return so we know if an expansion actually happened (i.e. an m-of-n
        # test was found)
        return wasexpanded

    def __draw_sample(self, nodename):
        """
        A function that takes the name of a node in the tree, and draws extra
        samples if fewer than the allowed minimum size of samples reach
        that node. (e.g. ifwe want 10,000 examples, and have 9,100, we will
        draw 900).
        The distributions are calculated with Gaussian Kernel Density Estimators
        and must be drawn in accordance with the relevant constraints in order
        for examples to reach the node.
        The list distrnodes is used and created by this function. This is a list
        of length equal to the number of features, and each entry is a node name,
        showing which node's feature distribution is currently being used to
        calculate that feature for this branch. (e.g. if we are on node '100',
        and the value for feature 3 is '1', we would check the current
        distribution of feature 3 against that for node '1', and if it is not
        different, use the distribution from feature 1 for sample drawing).
        """
        # Create a dictionary to hold the features for which only a single value
        # makes it through to this node. This feature will always take this value
        # as it's impossible to create a KDE for a single-valued distribution.
        singlevalsdict = {}
        # Find local samples and calculate how many are needed
        localsamples = self.traindata[self.tree[nodename]['reached']]
        samplesneeded = self.minsamples - localsamples.shape[0]
        # Find nodes used for kernel construction by parent node by feature
        # (used for checking if feature distributions have changed)
        parent = self.tree[nodename]['parent']
        if nodename == '1':
            distrs = ['1' for i in range(self.numfeats)]
        else:
            distrs = self.tree[parent]['distrnodes']
        # Pull out node constraints for use later
        constraints = self.tree[nodename]['constraints']
        # Create a distribution list for this node
        distrnodes = [nodename for i in range(self.numfeats)]
        # Set the bandwidth for the KDEs
        bandwidth = 1 / np.sqrt(localsamples.shape[0])
        # Only do any of this if we need samples
        if samplesneeded > 0:
            # Create list to store KDEs for each feature
            kernels = [None for i in range(self.numfeats)]
            for feat in range(self.numfeats):
                # Get the appropriate set of samples to check against
                distrindices = self.tree[distrs[feat]]['reached']
                parentsamples = self.traindata[distrindices]
                # Check if distribution for feature is diff from parent node
                # using Kolgomorov-Smirnov test
                # Including Bonferroni correction
                if stats.ks_2samp(localsamples[:, feat],
                                  parentsamples[:, feat])[1] <= self.significance / self.numfeats:
                    # Check for single values
                    uniques = np.unique(localsamples[:, feat])
                    if len(uniques) == 1:
                        singlevalsdict[feat] = uniques[0]
                        # If not single-valued, create KDE
                    else:
                        kernels[feat] = stats.gaussian_kde(localsamples[:, feat],
                                                           bw_method=bandwidth)
                else:
                    # If distribution doesn't differ, do same as above,
                    # but for parent node instead
                    uniques = np.unique(parentsamples[:, feat])
                    if len(uniques) == 1:
                        singlevalsdict[feat] = uniques[0]
                    else:
                        kernels[feat] = stats.gaussian_kde(parentsamples[:, feat],
                                                           bw_method=bandwidth)
                    # Record that we're still using the distribution from a node
                    # higher in the tree
                    distrnodes[feat] = distrs[feat]
            # Get the new samples, and append them to the local samples
            newsamples = self.__draw_instances(samplesneeded, kernels, constraints, singlevalsdict)
            allsamples = np.r_['0,2', localsamples, newsamples]
        # If we had enough, just return what we started with
        else:
            allsamples = localsamples
        # Set distribution of nodes as part of node values
        self.tree[nodename]['distrnodes'] = distrnodes
        return allsamples

    def __draw_instances(self, number, kernels, constraints, singlevalsdict):
        """
        Takes a number of samples to draw, a list of constraints (which consists
        of a list of m-of-n tests that should have been satisfied to reach this
        point, and a boolean variable saying whether the test should have been
        passed or failed) and a set of kernels to draw from, and produces a set
        of samples of that number using the kernels and constraints.
        This function calculates the conditional probabilities of each part of each
        m-of-n test being passed, then passes that information to a subfunction to
        draw the actual samples.
        """
        probslist = []
        feattests = {}
        # Loop over the m-of-n tests in the constraints list
        for test in constraints:
            probs = np.array([])
            probfeats = np.array([])
            # Pull out m and n
            m = test[0]
            n = len(test[1])
            # Check whether the test should have been passed
            if test[2]:
                # Loop over separate feature tests in the test
                for feattest in test[1]:
                    # Pull out items for clarity
                    feature = feattest[0]
                    threshold = feattest[1]
                    greater = feattest[2]
                    if feature in singlevalsdict:
                        # If the feature is a single value and guaranteed to be passed
                        # in the test, reduce m by one to account for this
                        # (Implied don't reduce m if not passed in test)
                        if (greater and singlevalsdict[feature] >= threshold) or (
                            (not greater) and singlevalsdict[feature] < threshold):
                            m -= 1
                    else:
                        # Get conditional probability of this feature passing test by integrating
                        # over kernel (to + or -inf depending on directino of test)
                        if greater:
                            conditional_prob = kernels[feature].integrate_box_1d(threshold, np.inf)
                        else:
                            conditional_prob = kernels[feature].integrate_box_1d(
                                -np.inf, threshold)
                        # Add this to the list of probabilities for this test
                        probs = np.append(probs, conditional_prob)
                        # And add the feature to the accopmanying list
                        probfeats = np.append(probfeats, feature)
                        # And add the feature to the dict of feature tests (better format
                        # for accessing them in later subfunction)
                        feattests[feature] = (threshold, greater)
                # Equalise probabilites to sum to 1 and add test to list
                probs = probs / sum(probs)
                testprobs = (m, probfeats, probs)
                probslist.append(testprobs)
            # In case the test should be failed, reverse the criteria and otherwise do
            # everything the same way (m becomes m-n)
            else:
                for feattest in test[1]:
                    feature = feattest[0]
                    threshold = feattest[1]
                    greater = not feattest[2]
                    if feature in singlevalsdict:
                        if (greater and singlevalsdict[feature] >= threshold) or (
                            (not greater) and singlevalsdict[feature] < threshold):
                            # m incremented instead of reduced because of reverse test
                            m += 1
                    else:
                        if greater:
                            conditional_prob = kernels[feature].integrate_box_1d(threshold, np.inf)
                        else:
                            conditional_prob = kernels[feature].integrate_box_1d(
                                -np.inf, threshold)
                        probs = np.append(probs, conditional_prob)
                        probfeats = np.append(probfeats, feature)
                        feattests[feature] = (threshold, greater)
                probs = probs / sum(probs)
                # We're doing reverse test, so need one more than n-m tests to be
                # passed to make sure m-of-n test is failed
                testprobs = (n + 1 - m, probfeats, probs)
                probslist.append(testprobs)
        # Call subfunction to draw instances
        instances = np.array([
            self.__draw_instance(kernels, probslist, feattests, singlevalsdict)
            for i in range(number)
        ])
        return instances

    def __draw_instance(self, kernels, testlist, feattests, singlevalsdict):
        """
        Takes a set of kernels, m-of-n tests expressed as conditional probabilities
        and feattests, a dictionary of tests keyed by feature. These are used to
        draw samples that pass the m-of-n tests in question, conditional on the
        probabilities of each n being part of the passed test.
        Singlevalsdict is a dictionary that tells us which features take only single
        values, and so need to be treated differently (no kernel exists for them).
        """
        def __passes_test(resample, test):
            """
            Simple helper function that takes a value and a single test as a (threshold,
            greater than) tuple, and checks whether the value passes.
            """
            passes = False
            if (test[1] and resample >= test[0]) or (not test[1] and resample < test[0]):
                passes = True
            return passes

        instance = np.zeros(self.numfeats)
        constrainedfeatures = []
        for conds in testlist:
            # Check we have conditions fo fulfil (i.e. all features in test are not
            # single-valued and so local m>0)
            if conds[0] > 0:
                # Choose weighted set of features from m-of-n test
                choices = np.random.choice(conds[1], p=conds[2], size=conds[0], replace=False)
                # Add those chosen to constraints
                constrainedfeatures = np.r_[constrainedfeatures, choices]
        for feature in range(self.numfeats):
            if feature in singlevalsdict:
                # If a single value, use that value
                instance[feature] = singlevalsdict[feature]
            elif feature not in constrainedfeatures:
                # Otherwise if not constrained draw a sample from the KDE
                instance[feature] = kernels[feature].resample(size=1)[0][0]
            else:
                # If constrained, draw samples from the KDE until one passes the
                # constraint
                found = False
                while not found:
                    resample = kernels[feature].resample(size=1)[0][0]
                    if __passes_test(resample, feattests[feature]):
                        found = True
                instance[feature] = resample
        return instance

    def __construct_test(self, nodename):
        """
        Takes a node, draws artifical samples for it, and finds the best
        m-of-n test to split on. Returns that test, plus the majority class
        labels for the samples that pass the test, and those that fail the test.
        """
        # Draw new samples
        if self.verbose:
            print("Drawing samples for node %s..." % nodename)
        newsamples = self.__draw_sample(nodename)
        # Get labels for them
        newlabels = np.array(self.oracle_predict(newsamples))
        if self.verbose:
            print("Finding thresholds...")
        tests = self.__make_candidate_tests(newsamples, newlabels, nodename)
        bestgain = 0
        # Find which test gives the best gain
        if self.verbose:
            print("Finding best test...")
        for test in tests:
            for threshold in tests[test]:
                testgain = self.__binary_info_gain(test, threshold, newsamples, newlabels)
                if testgain > bestgain:
                    bestgain = testgain
                    besttest = (test, threshold)
        # Find an m-of-n test as long as a best test was found. (If best test not
        # found, there was no info - return None)
        if bestgain > 0:
            if self.verbose:
                print("Finding m-of-n test...")
            mofntest = self.__make_mofn_tests(besttest, tests, newsamples, newlabels)
            # Get the list of samples that pass, and use to find majority classes
            # for samples that pass or that fail
            passes = [self.__passes_mn_test(sample, mofntest) for sample in newsamples]
            fails = [not i for i in passes]
            passclass = stats.mode(newlabels[passes]).mode[0]
            failclass = stats.mode(newlabels[fails]).mode[0]
        else:
            mofntest = None
            passclass = None
            failclass = None
        return (mofntest, passclass, failclass)

    def __make_candidate_tests(self, samples, labels, nodename):
        """
        Takes a set of samples and labels, and returns the possible breakpoints
        for each feature. These are the midpoints between any two samples that
        do not have the same label. A max of 20 will be returned for any one
        feature, and only breakpoints that lie within the range of real (rather
        than artificial) samples will be used.
        Breakpoints are stored as a dict with the feature being the key, and the
        values being the list of breakpoints.
        """
        # Create empty dictionary to store features and their breakpoints
        bpdict = {}
        # Get a list of the features that have already been used in this branch
        alreadyused = []
        for test in self.tree[nodename]['constraints']:
            for subtest in test[1]:
                alreadyused.append(subtest[0])
        # Loop over each feature
        for feature in range(samples.shape[1]):
            # Only generate breakpoints for a feature if it wasn't already used on this branch
            if feature not in alreadyused:
                # Get the minimum and maximum values from real examples
                featmin = min(self.traindata[self.tree[nodename]['reached']][:, feature])
                featmax = max(self.traindata[self.tree[nodename]['reached']][:, feature])
                # Get unique values for feature
                values = np.unique(samples[:, feature])
                breakpoints = []
                # Loop over values and check if diff classes between values
                for value in range(len(values) - 1):
                    # Check if different classes in associated labels, find midpoint if so
                    labels1 = labels[samples[:, feature] == values[value]]
                    labels2 = labels[samples[:, feature] == values[value + 1]]
                    l1unique = list(np.unique(labels1))
                    l2unique = list(np.unique(labels2))
                    if l1unique != l2unique or (l1unique == l2unique == [0, 1]):
                        midpoint = (values[value] + values[value + 1]) / 2
                        # If the point is within the range of real values, add it
                        # to breakpoints
                        if (midpoint > featmin) and (midpoint < featmax):
                            breakpoints.append(midpoint)
                # Trim list of breakpoints to 20 if too long
                if len(breakpoints) > 20:
                    idx = np.rint(np.linspace(0, len(breakpoints) - 1, num=20)).astype(int)
                    breakpoints = [breakpoints[i] for i in idx]
                # Add list of breakpoints to feature dict
                bpdict[feature] = breakpoints
        return bpdict

    def __entropy(self, labels):
        """
        Takes a list of labels, and calculates the entropy. Currently assumes binary
        labels of 0 and 1 - this would need to be altered if multiclass data has
        to be used.
        """
        # Check there are any labels to work with
        if len(labels) == 0:
            return 0
        prob = np.sum(labels) / len(labels)
        # Deal with the case where one class isn't present (would return nan without
        # this exception)
        if 0 < prob < 1:
            ent = -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
        else:
            ent = 0
        return ent

    def __binary_info_gain(self, feature, threshold, samples, labels):
        """
        Takes a feature and a threshold, examples and their
        labels, and find the best feature and breakpoint to split on to maximise
        information gain.
        Assumes only two classes. Would need to be altered if more are required.
        """
        # Get initial entropy
        origent = self.__entropy(labels)
        # Get two halves of threshold
        split1 = samples[:, feature] >= threshold
        split2 = np.invert(split1)
        # Get entropy after split (remembering to weight by no of examples in each
        # half of split)
        afterent = (self.__entropy(labels[split1]) * (np.sum(split1) / len(labels)) +
                    self.__entropy(labels[split2]) * (np.sum(split2) / len(labels)))
        gain = origent - afterent
        return gain

    def __mofn_info_gain(self, mofntest, samples, labels):
        """
        Takes an m-of-n test (see structure described in __create_node function)
        with a set of samples and labels, and calculates the information gain
        provided by that test.
        """
        # Unpack the tests structure
        m = mofntest[0]
        septests = mofntest[1]
        # List comprehension to generate a boolean index that tells us which samples
        # passed the test.
        splittest = np.array([
            samples[:, septest[0]] >= septest[1]
            if septest[2] else samples[:, septest[0]] < septest[1] for septest in septests
        ])
        # Now check whether the number of tests passed per sample is higher than m
        split1 = np.sum(splittest, axis=0) >= m
        split2 = np.invert(split1)
        # Calculate original entropy
        origent = self.__entropy(labels)
        # Get entropy of split
        afterent = (self.__entropy(labels[split1]) * (np.sum(split1) / len(labels)) +
                    self.__entropy(labels[split2]) * (np.sum(split2) / len(labels)))
        gain = origent - afterent
        return gain

    def __expand_mofn_test(self, test, feature, threshold, greater, incrementm):
        """
        Constructs and returns a new m-of-n test using the passed test and
        other parameters. These are the feature, the threshold to split on,
        whether we take items less than or greater than (or equal to) the
        threshold, and whether or not to increment m (i.e. do we change from an
        m-of-n test to an m-of-n+1 test, or an m+1-of-n+1 test).
        """
        # Create the test
        if incrementm:
            newm = test[0] + 1
        else:
            newm = test[0]
        newfeats = list(test[1])
        newfeats.append((feature, threshold, greater))
        newtest = (newm, newfeats)
        return newtest

    def __make_mofn_tests(self, besttest, tests, samples, labels):
        """
        Finds the best m-of-n test, using a beam width of 2, by looping over
        all possible thresholds and operators. Only tests which are better
        (by a percentage determined by self.improvement) than the current best
        test are kept, and only the best 2 of these are kept from each round of
        addition of features.
        """
        # Initialise beam with best test and its negation
        initgain = self.__binary_info_gain(besttest[0], besttest[1], samples, labels)
        beam = [(1, [(besttest[0], besttest[1], False)]), (1, [(besttest[0], besttest[1], True)])]
        beamgains = [initgain, initgain]
        # Initialise current beam (which will be modified within the loops)
        currentbeam = list(beam)
        currentgains = list(beamgains)
        beamchanged = True
        # Set up loop to repeat until beam isn't changed
        while beamchanged:
            # Save the current best gain for comparison
            currentbest = np.max(currentgains)
            beamchanged = False
            # Loop over the current best m-of-n tests in beam
            for test in beam:
                # Get features used in the test already
                existingfeats = [subtest[0] for subtest in test[1]]
                # Loop over the single-features in candidate tests dict
                for feature in tests:
                    # Check it hasn't been used in the test already
                    if feature not in existingfeats:
                        # Loop over the thresholds for the feature
                        for threshold in tests[feature]:
                            # Loop over greater than/lesser than tests
                            for greater in [True, False]:
                                # Loop over m+1-of-n+1 and m-of-n+1 tests
                                for incrementm in [True, False]:
                                    # Add selected feature+threshold to to current test
                                    newtest = self.__expand_mofn_test(
                                        test, feature, threshold, greater, incrementm)
                                    # Get info gain and compare it
                                    gain = self.__mofn_info_gain(newtest, samples, labels)
                                    # Compare gains
                                    if (gain > min(currentgains)) \
                                       and (gain > self.improvement * currentbest):
                                        # Replace worst in beam if gain better than worst in beam
                                        currentbeam[np.argmin(currentgains)] = newtest
                                        currentgains[np.argmin(currentgains)] = gain
                                        beamchanged = True
            # Set new tests in beam and associated gains
            beam = list(currentbeam)
            beamgains = list(currentgains)
        # Return the best test in beam
        return beam[np.argmax(beamgains)]

    def oracle_predict(self, examples):
        """
        Returns predictions from the oracle from a set of samples (as a numpy
        array). Currently only works with sklearn MLPClassifiers, but can easily
        be expanded to handle other oracles too.
        """
        predictions = []
        # If only one example
        if len(np.array(examples).shape) == 1:
            predictions = [self.oracle.predict(examples.reshape(1, -1))]
        else:
            predictions = [self.oracle.predict(example.reshape(1, -1))[0] for example in examples]
        return predictions

    def __prune_tree(self):
        """
        Where two leaf nodes have the same parent and make the same prediction,
        they are removed and the parent becomes a leaf. Done with a while loop
        that resets each time the tree is pruned, as otherwise difficult to
        keep track of looping over all nodes while being pruned.
        """
        altered = True
        while altered:
            altered = False
            # Get list of keys to avoid iterating over dict while editing it
            nodes = list(self.tree.keys())
            for node in nodes:
                # Find sister
                sister = self.tree[node]['sister']
                # If classes the same, remove both nodes and make parent leaf
                if sister \
                   and self.tree[node]['isleaf'] \
                   and self.tree[sister]['isleaf'] \
                   and self.tree[node]['predictedclass'] \
                   == self.tree[sister]['predictedclass']:
                    parent = self.tree[node]['parent']
                    self.tree[parent]['isleaf'] = True
                    self.tree[parent]['mntest'] = None
                    self.tree[parent]['0daughter'] = None
                    self.tree[parent]['1daughter'] = None
                    del self.tree[node]
                    del self.tree[sister]
                    altered = True
                    break

    def __predict_sample(self, sample):
        """
        Predicts the class of a single sample using a TREPAN tree. Checks whether
        the sample pases the m-of-n test for each node and traverses the tree until
        it finds a leaf.
        """
        # Check sample length is correct
        if len(sample) != self.numfeats:
            print("Sample has incorrect number of features")
            return
        # Start at the root node
        node = '1'
        # Traverse the tree until leaf found
        while not self.tree[node]['isleaf']:
            passed = self.__passes_mn_test(sample, self.tree[node]['mntest'])
            if passed:
                node = self.tree[node]['1daughter']
            else:
                node = self.tree[node]['0daughter']
        # Return class of the leaf node reached
        return self.tree[node]['predictedclass']

    def predict(self, samples):
        """
        Predicts the class of a list of samples provided, returned as a list of the
        same length.
        """
        # Check we fitted the tree
        if not (self.fitted or self.building):
            raise Exception('Tree must be fitted before applying predictive methods.')
        # Check for single sample
        if len(np.array(samples).shape) == 1:
            classes = [self.__predict_sample(samples)]
        else:
            classes = [self.__predict_sample(sample) for sample in samples]
        return classes

    def accuracy(self, samples, labels):
        """
        Predicts for a set of samples using a TREPAN tree, and calculates the
        percentage of the time they match a provided set of labels.
        """
        # Check we fitted the tree
        if not (self.fitted or self.building):
            raise Exception('Tree must be fitted before applying predictive methods.')
        predictions = self.predict(samples)
        matches = predictions == labels
        accuracy = sum(matches) / len(matches)
        return accuracy

    def fidelity(self, samples):
        """
        Predicts for a set of samples using a TREPAN tree, and calculates the
        percentage of the time that they match with the prediction of the oracle
        used to build it.
        """
        # Check we fitted the tree
        if not (self.fitted or self.building):
            raise Exception('Tree must be fitted before applying predictive methods.')
        treepredictions = np.array(self.predict(samples))
        oraclepredictions = self.oracle_predict(samples)
        matches = oraclepredictions == treepredictions
        fidelity = sum(matches) / len(matches)
        return fidelity

    def draw_tree(self, filename):
        """
        Produces a .dot file representing the tree built by a TREPAN run, and saves
        it with the provided filename.
        This currently loops through the relevant dictionary and adds nodes one at
        a time to the .dot file - this means that nodes will not be added in any
        specific order. This should not matter for usage, as .dot files do not
        require any enforcement of graph order, but does make the output files
        a bit less readable.
        """
        def write_test(test):
            """
            Small subfunction to write the m-of-n test as a string.
            """
            # Initialise string and add the m
            output = "[color=blue, label=\"%d of\\n\\n" % test[0]
            # Add the tests one by one
            for subtest in test[1]:
                # Define test symbol
                if subtest[2]:
                    substr = "%s\\n " % (self.featnames[subtest[0]])
                    # symbol = ">="
                else:
                    substr = "¬%s\\n " % (self.featnames[subtest[0]])
                    # symbol = "<"
                # Create string and add it to existing string
                # substr = "%s %s %f\\n " % (self.featnames[subtest[0]], symbol,
                #                            subtest[1])
                output = output + substr
            # Close string and return it
            output = output + "\", shape=box]"
            return output

            # Check we fitted the tree

        if not self.fitted:
            raise Exception('Tree must be fitted before drawing.')
        # Open file and write to it
        with open(filename, 'w') as out_hndl:
            # Initiate graph
            out_hndl.write("digraph tree\n{\n")
            # Add nodes one by one
            for node in self.tree:
                # Check if it's a leaf node: if no, show m-of-n test
                if not self.tree[node]['isleaf']:
                    # Write test
                    out_hndl.write("\t%s %s;\n" % (node, write_test(self.tree[node]['mntest'])))
                    # Write links
                    out_hndl.write("\t%s -> %s [label=\"False\"];\n" %
                                   (node, self.tree[node]['0daughter']))
                    out_hndl.write("\t%s -> %s [label=\"True\"];\n" %
                                   (node, self.tree[node]['1daughter']))
                # Otherwise, show predicted class
                else:
                    # out_hndl.write("\t%s [color=blue, fontcolor=black, label=\"%f\"];\n" %
                    #                (node, self.tree[node]['predictedclass']))
                    out_hndl.write("\t%s [color=blue, fontcolor=black, label=\"%r\"];\n" %
                                   (node, self.tree[node]['predictedclass'][0] > 0.5))
            # Close graph
            out_hndl.write("}\n")

    def score(self, samples, labels):
        return self.accuracy(samples, labels)
