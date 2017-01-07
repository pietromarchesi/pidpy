# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 00:22:10 2016

@author: Tycho


mini documentation:
nah fuck documentation. #yolowek

"""

import numpy as np
from PID2_computations import *

def entropy(probs):
    
    '''
    For given prob vector returns the entropy.
    '''
    
    if abs(np.sum(probs)-1.0) > 10**(-5): #what cutoff to choose?
        raise ValueError('probability vector must be normalised')
        
    entropy = 0 
    for i in xrange(len(probs)):
        entropy += -1 * probs[i] * np.log2(probs[i])
        
    #if we get nan set to zero
    if np.isnan(entropy):
        entropy = 0
    return entropy


def mutual_info(prior_probs_y, prior_probs_x, joint_probs):
    '''
    MI function in terms of joint probability. First prior probability vector
    corresponds to the first index in the joint_prob matrix.
    '''
    #print prior_probs_y    
    
    if abs(np.sum(prior_probs_y)-1.0) > 10**(-5): #what cutoff to choose?
        raise ValueError('prior_probs_y must be normalised')
    
    if abs(np.sum(prior_probs_x)-1.0) > 10**(-5): #what cutoff to choose?
        raise ValueError('prior_probs_x must be normalised')
    
    if abs(np.sum(joint_probs)-1.0) > 10**(-5): #what cutoff to choose?
        raise ValueError('joint_probs must be normalised')
    
    MI = 0
    for i in xrange(len(prior_probs_y)):
        #SI = 0
        for j in xrange(len(prior_probs_x)):
            if (abs(joint_probs[i,j]) > 10**(-8)):
                MI += joint_probs[i,j] * np.log2(1.0 * joint_probs[i,j]/(prior_probs_y[i]*prior_probs_x[j]))
#            else:
#                print "ive found a zero!"

    return MI


def synergy(MI, redundancy, *args):
    '''
    Computes the synergistic information two sets of variables hold over another
    *args = variable number of unique information sources, all need to be provided
    '''    
    
    total_uniques = np.sum(args)
    synergy = MI - redundancy - total_uniques
    return synergy

def unique(MI, redundancy):
    '''    
    Computes the unique information that a variable has given the mutual information
    between that variable and target and the redundant information with the other source
    '''
    unique = MI - redundancy
    return unique

def spec_info(prior, pxy, pyx):
    '''
    Computes the specific information of source wrt a specific outcome of the 
    target variable. Sum is over the different outcomes of the source variable.
    Thus:
    Elements of the arrays have to match up for each x-value.
    '''
    
    si = 0
    for i in xrange(len(pxy)):
        contribution = pxy[i] * (np.log2(1.0/prior) - np.log2(1.0/pyx[i]))
        
        #set nans to zero
        if np.isnan(contribution):
            contribution = 0
            
        si += contribution
    return si

#finish this function and think of how to compute the jointprob 
#of one hidden neuron and the target 
#def spec_info_jointprob(y,jointprob):
#    '''
#    Computes the specific information of binary source wrt a specific outcome
#    of the target variable.
#    
#    Input:
#    y: integer, specifies the outcome of the target variable (between 0 and 9)
#    jointprob: 10x2 matrix, each row: [y0,y1] format.
#    '''
#    #prior target outcome probabilities
#    priors_y = np.sum(jointprob,1)
#    #pick out the appropriate value
#    prior_y = priors_y[y]
#    
#    #prior source outcome probabilities
#    priors_x = np.sum(jointprob,0)    
#    
#    #assume binary source variable 
#    Ispec = 0
#    
#    if (abs(jointprob[y,0]) > 10**(-10)):   
#        Ispec += 1.0/prior_y * jointprob[y,0] * np.log2(1.0*jointprob[y,0]/(prior_y*priors_x[0]))
#    if (abs(jointprob[y,1]) > 10**(-10)):
#        Ispec += 1.0/prior_y * jointprob[y,1] * np.log2(1.0*jointprob[y,1]/(prior_y*priors_x[1]))
#
#    #check if Ispec is a NaN and set to zero if it is
#    if Ispec != Ispec:
#        Ispec = 0
#    
#    return Ispec
    

    
def redundancy_2(priors_y, priors_x1, priors_x2, joint_yx1, joint_yx2):
    '''
    Redundancy calculation based on simplifications (see thesis folder). 
    Two sources for simplicity, can be implemented with only the jointprob with extra effort
    We assume binary outcomes for each source
    '''    
    
    red = 0
    #loop over the y values!
    for i in xrange(joint_yx1.shape[0]):
        cont1 = 0
        cont2 = 0

        if (abs(joint_yx1[i,0]) > 10**(-10)):
            cont1 = joint_yx1[i,0]*np.log2(1.0*joint_yx1[i,0]/(1.*priors_y[i]*priors_x1[0]))
        if (abs(joint_yx1[i,1]) > 10**(-10)): 
            cont1 += joint_yx1[i,1]*np.log2(1.0*joint_yx1[i,1]/(1.*priors_y[i]*priors_x1[1]))
        
        if (abs(joint_yx2[i,0]) > 10**(-10)):
            cont2 = joint_yx2[i,0]*np.log2(1.0*joint_yx2[i,0]/(1.*priors_y[i]*priors_x2[0])) 
        if (abs(joint_yx2[i,1]) > 10**(-10)): 
            cont2 += joint_yx2[i,1]*np.log2(1.0*joint_yx2[i,1]/(1.*priors_y[i]*priors_x2[1]))
        red += min(cont1,cont2)
        #print 'cont1: %f' % cont1
        #print 'cont2: %f' % cont2
    #print 'red: %f' % red
    return red
    
#Imax is defined exactly like Imin but then replacing min with max (thesis williams)
def Imax2(priors_y, priors_x1, priors_x2, joint_yx1, joint_yx2):
    imax = 0
    #loop over the y values!
    for i in xrange(joint_yx1.shape[0]):
        cont1 = 0
        cont2 = 0

        if (abs(joint_yx1[i,0]) > 10**(-10)):
            cont1 = joint_yx1[i,0]*np.log2(1.0*joint_yx1[i,0]/(1.*priors_y[i]*priors_x1[0]))
        if (abs(joint_yx1[i,1]) > 10**(-10)): 
            cont1 += joint_yx1[i,1]*np.log2(1.0*joint_yx1[i,1]/(1.*priors_y[i]*priors_x1[1]))
        
        if (abs(joint_yx2[i,0]) > 10**(-10)):
            cont2 = joint_yx2[i,0]*np.log2(1.0*joint_yx2[i,0]/(1.*priors_y[i]*priors_x2[0])) 
        if (abs(joint_yx2[i,1]) > 10**(-10)): 
            cont2 += joint_yx2[i,1]*np.log2(1.0*joint_yx2[i,1]/(1.*priors_y[i]*priors_x2[1]))
        imax += max(cont1,cont2)
    return imax

#PID2:
#partial information decomposition for 2 sources and target variable

#we need to calculate the mutual information terms (3), and the redundancy
#the rest follows. For the mutual information terms, we need:

def red(priors, px1y, pyx1, px2y, pyx2):
    '''
    Redundancy calculation based on the conditional distributions, which calls
    the specific information function.
    Before using, recheck that everything works as it should.
    
    We assume only two sources and binary outcomes for each source.
    '''    
    
    red = 0
    #here i loops over y-values
    for i in xrange(len(priors)):
        red += priors[i] * min(spec_info(priors[i], px1y[i,:], pyx1[i,:]), \
            spec_info(priors[i], px2y[i,:], pyx2[i,:]))
    return red




def PID_2(jointprob):
    '''
    PID2 using joint probabilities
    this only works for 2 sources since I'm picking out the appropriate values from
    the joint probability matrix which goes like:
    [[p(y=0,x1=0,x2=0), p(y=0,x1=1,x2=0), etc],[p(y=1,x1=0,x2=0), etc], etc]
    so first index is wrt the target variable
    '''
    
    #from the jointprob matrix, compute all the necessary things to give to the functions
    #print jointprob
    
    priors_y = np.sum(jointprob,1)
    priors_x12 = np.sum(jointprob,0)
    priors_x1 = np.array([priors_x12[0]+priors_x12[2], priors_x12[1]+priors_x12[3]])
    priors_x2 = np.array([priors_x12[0]+priors_x12[1], priors_x12[2]+priors_x12[3]])
    #this could maybe be shorter:
    
    joint_yx1, joint_yx2 = [np.zeros([jointprob.shape[0],2]), np.zeros([jointprob.shape[0],2])]
    joint_yx1[:,0] = jointprob[:,0] + jointprob[:,2]
    joint_yx1[:,1] = jointprob[:,1] + jointprob[:,3]
    joint_yx2[:,0] = jointprob[:,0] + jointprob[:,1]
    joint_yx2[:,1] = jointprob[:,2] + jointprob[:,3]
    
    #compute mutual info between {X1,X2} and Y        
    MI12 = mutual_info(priors_y, priors_x12, jointprob)
    #print 'MI12: %f' % MI12
    
    #compute mutual info between target and seperate sources
    MI1 = mutual_info(priors_y, priors_x1, joint_yx1)
    MI2 = mutual_info(priors_y, priors_x2, joint_yx2)  
    #print 'MI1: %f' % MI1
    
    #compute redundancy (i.e. shared info) between 2 sources
    #redundancy = red(priors_y, condprob_x1y, condprob_yx1, condprob_x2y, condprob_yx2)
    redundancy = redundancy_2(priors_y, priors_x1, priors_x2, joint_yx1, joint_yx2)
    
    #compute unique info by deducting redundancy from indiv mutual info terms    
    unique_x1 = unique(MI1,redundancy)
    unique_x2 = unique(MI2,redundancy)
    
    #compute synergy by deducting all other PID terms from mutual info between
    #both sources together and target
    syn = synergy(MI12, redundancy, unique_x1, unique_x2)
    
    return redundancy, unique_x1, unique_x2, syn, MI1, MI2, MI12  


def Synergy_computer_Imax(jointprob):
    priors_y = np.sum(jointprob,1)
    priors_x12 = np.sum(jointprob,0)
    priors_x1 = np.array([priors_x12[0]+priors_x12[2], priors_x12[1]+priors_x12[3]])
    priors_x2 = np.array([priors_x12[0]+priors_x12[1], priors_x12[2]+priors_x12[3]])
    #this could maybe be shorter:
    
    joint_yx1, joint_yx2 = [np.zeros([jointprob.shape[0],2]), np.zeros([jointprob.shape[0],2])]
    joint_yx1[:,0] = jointprob[:,0] + jointprob[:,2]
    joint_yx1[:,1] = jointprob[:,1] + jointprob[:,3]
    joint_yx2[:,0] = jointprob[:,0] + jointprob[:,1]
    joint_yx2[:,1] = jointprob[:,2] + jointprob[:,3]
    
    #compute mutual info between {X1,X2} and Y        
    MI12 = mutual_info(priors_y, priors_x12, jointprob)
    #print 'MI12: %f' % MI12
    
    #compute mutual info between target and seperate sources
    #MI1 = mutual_info(priors_y, priors_x1, joint_yx1)
    #MI2 = mutual_info(priors_y, priors_x2, joint_yx2)  
    
    imax = Imax2(priors_y, priors_x1, priors_x2, joint_yx1, joint_yx2)
    
    syn = MI12 - imax
    return syn
    

def redundancy_2_vars(probability_1_target,probability_2_target):
    '''
    redundancy function that works for combintions of multiple (binary) variables!    
    Input:
    probability_1_target = joint probability function of the target (rows) and the first source (can be multiple variables)
    probability_2_target = ---- same but for second source ---- size = [10,2 ^ nr vars]
    
    tests:
    in the case where source 1 and source 2 are simple binary values, compare with existing PID functions
    self redundancy = mutual information
    
    '''
    
    if probability_1_target.shape[0] != probability_2_target.shape[0]:
        raise ValueError('target needs to have the same nr of outcomes')
    
    priors_y1 = np.sum(probability_1_target,1)
    priors_y2 = np.sum(probability_2_target,1)
    
    if np.sum(abs(priors_y1-priors_y2)) > 10**(-8):
        raise ValueError('somethings wrong with the distributions. the priors of y do not match')
    
    redundancy = 0
    for i in xrange(probability_1_target.shape[0]):
        redundancy += priors_y1[i] * min(spec_info_jointprob(i,probability_1_target),spec_info_jointprob(i,probability_2_target))
    
    return redundancy
    
def redundancy_3_vars(probability_1_target,probability_2_target,probability_3_target):
    '''
    redundancy function that works for combintions of multiple (binary) variables!    
    Input:
    probability_1_target = joint probability function of the target (rows) and the first source (can be multiple variables)
    probability_2_target = ---- same but for second source ---- size = [10,2 ^ nr vars]
    
    tests:
    in the case where source 1 and source 2 are simple binary values, compare with existing PID functions
    self redundancy = mutual information
    
    '''
    
    if probability_1_target.shape[0] != probability_2_target.shape[0]:
        raise ValueError('target needs to have the same nr of outcomes')
    if probability_1_target.shape[0] != probability_3_target.shape[0]:
        raise ValueError('target needs to have the same nr of outcomes')   
    
    
    priors_y1 = np.sum(probability_1_target,1)
    priors_y2 = np.sum(probability_2_target,1)
    priors_y3 = np.sum(probability_3_target,1)
    
    if np.sum(abs(priors_y1-priors_y2)) > 10**(-8):
        raise ValueError('somethings wrong with the distributions. the priors of y do not match')
    if np.sum(abs(priors_y1-priors_y3)) > 10**(-8):
        raise ValueError('somethings wrong with the distributions. the priors of y do not match')
    

    redundancy = 0
    for i in xrange(probability_1_target.shape[0]):
        redundancy += priors_y1[i] * min(spec_info_jointprob(i,probability_1_target), \
                    spec_info_jointprob(i,probability_2_target), spec_info_jointprob(i,probability_3_target))
    
    return redundancy


def Imax_3_vars(probability_1_target,probability_2_target,probability_3_target):
    '''
    Imax function that works for combintions of multiple (binary) variables!    
    Input:
    probability_1_target = joint probability function of the target (rows) and the first source (can be multiple variables)
    probability_2_target = ---- same but for second source ---- size = [10,2 ^ nr vars]
    
    tests:
    in the case where source 1 and source 2 are simple binary values, compare with existing PID functions
    self redundancy = mutual information
    U1, U2, U3, R1_2_3, S123, Remaining_terms, MI1, MI2, MI3, MI123
    '''
    
    if probability_1_target.shape[0] != probability_2_target.shape[0]:
        raise ValueError('target needs to have the same nr of outcomes')
    if probability_1_target.shape[0] != probability_3_target.shape[0]:
        raise ValueError('target needs to have the same nr of outcomes')
    
    priors_y1 = np.sum(probability_1_target,1)
    priors_y2 = np.sum(probability_2_target,1)
    priors_y3 = np.sum(probability_3_target,1)
    
    if np.sum(abs(priors_y1-priors_y2)) > 10**(-8):
        raise ValueError('somethings wrong with the distributions. the priors of y do not match')
    if np.sum(abs(priors_y1-priors_y3)) > 10**(-8):
        raise ValueError('somethings wrong with the distributions. the priors of y do not match')

    
    Imax = 0
    for i in xrange(probability_1_target.shape[0]):
        Imax += priors_y1[i] * max(spec_info_jointprob(i,probability_1_target), \
                spec_info_jointprob(i,probability_2_target), spec_info_jointprob(i,probability_3_target))
    
    return Imax

#def Imax_general(*argv):
    '''
    Maybe you want to make this a list with probability distributions!
    '''


def spec_info_jointprob(y, jointprob):
    '''
    Computes the specific info of source wrt a specific outcome of the target variable.
    Does not assume a single binary source variable, but can do 2.

    Input:
    y: integer, specifies the outcome of the target variable (between 0 and 9)
    jointprob: 10x2 matrix, each row: [y0,y1] format.
    '''

    priors_y = np.sum(jointprob, 1)
    prior_y = priors_y[y]

    # prior source outcome probabilities
    priors_x = np.sum(jointprob, 0)

    Ispec = 0

    for i in xrange(jointprob.shape[1]):
        if (abs(jointprob[y, i]) > 10 ** (-10)):
            Ispec += 1.0 / prior_y * jointprob[y, i] * np.log2(
                1.0 * jointprob[y, i] / (prior_y * priors_x[i]))

    if Ispec != Ispec:
        Ispec = 0

    return Ispec


def Imax_general(prob_dists):
    '''
    input: prob_dists is a list of probability distributions each one being
    a prob dist between one of the sources to be considered and the target.    
    
    Computes the Imax following definition using the spec_info_jointprob function.
    For each state of the target it computes the specific infos of each of the
    sources (for which the prob_dist is given). Then takes the max value out of
    that list. This is done for each of the (10) target states. The values are
    summed and outputted as Imax.
    '''
    
#    prob_dists = []
#    for i in argv:
#        prob_dists.append(i)
        
    #the prior target probs should be the same for all the prob distributions
    #you could test this by raising value error if not.
#    priors_y = np.zeros(len(prob_dists))
#    
#    for i in xrange(len(prob_dists)):
#        priors_y[i] = np.sum(prob_dists[i],1)
    
    priors_y = np.sum(prob_dists[0],1)
    
    Imax = 0 
    for j in xrange(len(priors_y)):
        spec_infos = [spec_info_jointprob(j,prob_dists[i]) for i in xrange(len(prob_dists))]
        Imax += priors_y[j] * max(spec_infos)
    
    return Imax

def Imin_general(prob_dists):
    '''
    Same as above for the Imax_general function but then instead of maximum 
    the minimum is taken of the list of spec_info values.
    '''
    priors_y = np.sum(prob_dists[0],1)
    
    Imin = 0 
    for j in xrange(len(priors_y)):
        spec_infos = [spec_info_jointprob(j,prob_dists[i]) for i in xrange(len(prob_dists))]
        Imin += priors_y[j] * min(spec_infos)
    
    return Imin

def Overall_synergy(labels,hidden_states,selection):
    '''
    Function that computes the total synergy between the hidden neurons in the selection
    based on the fact that
    synergy = I(target:all_neurons) - Imax(target:all_one_order_lower selections)
    
    Requires significant computational power because it needs to compute 
    len(selection) + 1 probability distributions that can be large. Namely
    1 of size 10x2^(len(selection)) and 
    len(selection) of size 10x2^(len(selection)-1)
    '''
    
    total_prob = compute_jointprob_full(labels, hidden_states, selection)
    priors_y = np.sum(total_prob, 1)
    priors_tot = np.sum(total_prob, 0)
        
    MItot = mutual_info(priors_y,priors_tot,total_prob)
    
    marginals_one_order_lower = []
#    priors_marginals = []
    for i in selection:
        selection_minus_i = [l for l in selection if l != i]
        #print selection_minus_i
        marginal = compute_jointprob_full(labels, hidden_states, selection_minus_i)
        marginals_one_order_lower.append(marginal)
#        priors_marginals.append(np.sum(marginal,0))
    #print len(marginals_one_order_lower)
    
    Imax = Imax_general(marginals_one_order_lower)
    
    synergy = MItot - Imax
    #return marginals_one_order_lower
    return synergy#, MItot, Imax#, marginals_one_order_lower
    
def Overall_synergy_and_MItot(labels,hidden_states,selection):
    '''
    Function that computes the total synergy between the hidden neurons in the selection
    based on the fact that
    synergy = I(target:all_neurons) - Imax(target:all_one_order_lower selections)
    
    Requires significant computational power because it needs to compute 
    len(selection) + 1 probability distributions that can be large. Namely
    1 of size 10x2^(len(selection)) and 
    len(selection) of size 10x2^(len(selection)-1)
    '''
    
    total_prob = compute_jointprob_full(labels, hidden_states, selection)
    priors_y = np.sum(total_prob, 1)
    priors_tot = np.sum(total_prob, 0)
        
    MItot = mutual_info(priors_y,priors_tot,total_prob)
    
    marginals_one_order_lower = []
#    priors_marginals = []
    for i in selection:
        selection_minus_i = [l for l in selection if l != i]
        #print selection_minus_i
        marginal = compute_jointprob_full(labels, hidden_states, selection_minus_i)
        marginals_one_order_lower.append(marginal)
#        priors_marginals.append(np.sum(marginal,0))
    #print len(marginals_one_order_lower)
    
    Imax = Imax_general(marginals_one_order_lower)
    
    synergy = MItot - Imax
    #return marginals_one_order_lower
    return synergy, MItot#, Imax#, marginals_one_order_lower

def Overall_redundancy(labels, hidden_states, selection):
    '''
    Function that computes the total redundancy of all the neurons in the selection.
    For each of the neurons in the selection computes the marginals between neuron 
    and target and appends the distribution to a list of all marginals. 
    Then computes Imin of all sources using the Imin function that accepts a 
    list of prob distributions.
    '''
    
    
    prob_dists = []
    for i in xrange(len(selection)):
        #mja = selection[i]
        prob_dister = compute_jointprob_single_neuron(labels, hidden_states, selection[i])
        prob_dists.append(prob_dister)
    
    Imin = Imin_general(prob_dists)
    
    return Imin #, prob_dister, labels, hidden_states, mja
    
def Unique_contribution(labels, hidden_states, selection, picked_neuron):
    '''
    Unique contribution of the 'picked_neuron' is given based on the fact that
    the unique contribution is 
    I(target:selected_neuron) - intersection(I(target:selected_neuron) and I(target:selection\selected_neuron))
    where intersection = Imin!
     => CAREFUL:picked_neuron is not the index of the selection of neurons BUT the actual neuron number.
    so if selection = [0,2,5], then picked_neuron = 2 means that neuron number 2 is picked, NOT neuron number 5!
    '''
    #you could test here whether the neuron that is given is actually in the selection!
    
    marginal_picked_neuron = compute_jointprob_single_neuron(labels, hidden_states, picked_neuron)
    marginal_target = np.sum(marginal_picked_neuron,1)
    marginal_source = np.sum(marginal_picked_neuron,0)
    
    MIpicked_neuron = mutual_info(marginal_target,marginal_source,marginal_picked_neuron)
    
    selection_minus_picked = [l for l in selection if l != picked_neuron]
    marginal_rest = compute_jointprob_full(labels, hidden_states, selection_minus_picked)
    #you could recompute the marginal_target and compare just to be sure.
    
    Imin = Imin_general([marginal_picked_neuron,marginal_rest])
    
    Unique_cont = MIpicked_neuron - Imin
    
    return Unique_cont

def Unique_contribution_and_MI(labels, hidden_states, selection, picked_neuron):
    '''
    Unique contribution of the 'picked_neuron' is given based on the fact that
    the unique contribution is 
    I(target:selected_neuron) - intersection(I(target:selected_neuron) and I(target:selection\selected_neuron))
    where intersection = Imin!
     => CAREFUL:picked_neuron is not the index of the selection of neurons BUT the actual neuron number.
    so if selection = [0,2,5], then picked_neuron = 2 means that neuron number 2 is picked, NOT neuron number 5!
    '''
    #you could test here whether the neuron that is given is actually in the selection!
    
    marginal_picked_neuron = compute_jointprob_single_neuron(labels, hidden_states, picked_neuron)
    marginal_target = np.sum(marginal_picked_neuron,1)
    marginal_source = np.sum(marginal_picked_neuron,0)
    
    MIpicked_neuron = mutual_info(marginal_target,marginal_source,marginal_picked_neuron)
    
    
    
    selection_minus_picked = [l for l in selection if l != picked_neuron]
    marginal_rest = compute_jointprob_full(labels, hidden_states, selection_minus_picked)
    #you could recompute the marginal_target and compare just to be sure.
    
    Imin = Imin_general([marginal_picked_neuron,marginal_rest])
    
    Unique_cont = MIpicked_neuron - Imin
    
    return Unique_cont, MIpicked_neuron

    
def Overall_unique_contributions(labels, hidden_states, selection):
    '''
    Function that computes the unique contributions of all the neurons in the selection. 
    Requures significant computational power because function needs to compute a
    2*len(selection) number of probability tables, len(selection) of which can be large
    namely 2^(len(selection-1))
    
    input = labels, hidden_states, and a selection of neurons: 'selection'
    output = unique info contributions for each of the neurons of the selection:
    THE ORDER IS PRESERVED IN THE OUTPUT ARRAY, so if
    selection = [0,3,8], that means that
    output = [U0, U3, U8]
    '''
    
    unique_contributions = np.zeros(len(selection))
    for i in xrange(len(selection)):
        unique_contributions[i] = Unique_contribution(labels, hidden_states, selection, selection[i])
        if abs(unique_contributions[i]) < 10**(-10):
            unique_contributions[i] = 0
        
        
    return unique_contributions
    
def Overall_unique_contributions_and_MI(labels, hidden_states, selection):
    '''
    Function that computes the unique contributions of all the neurons in the selection. 
    Requures significant computational power because function needs to compute a
    2*len(selection) number of probability tables, len(selection) of which can be large
    namely 2^(len(selection-1))
    
    input = labels, hidden_states, and a selection of neurons: 'selection'
    output = unique info contributions for each of the neurons of the selection:
    THE ORDER IS PRESERVED IN THE OUTPUT ARRAY, so if
    selection = [0,3,8], that means that
    output = [U0, U3, U8]
    '''
    
    unique_contributions = np.zeros(len(selection))
    MI_ind_neurons = np.zeros(len(selection))
    for i in xrange(len(selection)):
        unique_contributions[i],MI_ind_neurons[i] = Unique_contribution_and_MI(labels, hidden_states, selection, selection[i])
        if abs(unique_contributions[i]) < 10**(-10):
            unique_contributions[i] = 0
        if abs(MI_ind_neurons[i]) < 10**(-10):
            MI_ind_neurons[i] = 0
        
    return unique_contributions,MI_ind_neurons



def PID3_terms(tripletprob):
    '''
    This function computes for a given prob function of 3 binary sources and the target
    The following PID2 terms: U1, U2, U3, R123, S123.
    And outputs the remaining terms as 'remaining'
    
    Input is a 10 by 8 prob table where rows = target &
    columns = sources with the order 000,100,010,001,110,101,011,111
    
    
    Output:
    
    '''
    
    #compute the marginal probabilities by appropriate summing
    priors_y = np.sum(tripletprob,1)
    priors_x123 = np.sum(tripletprob,0)
    priors_x1 = np.array([priors_x123[0]+priors_x123[2]+priors_x123[3]+priors_x123[6], priors_x123[1]+priors_x123[4]+priors_x123[5]+priors_x123[7]])
    priors_x2 = np.array([priors_x123[0]+priors_x123[1]+priors_x123[3]+priors_x123[5], priors_x123[2]+priors_x123[4]+priors_x123[6]+priors_x123[7]])
    priors_x3 = np.array([priors_x123[0]+priors_x123[1]+priors_x123[2]+priors_x123[4], priors_x123[3]+priors_x123[5]+priors_x123[6]+priors_x123[7]])
    
    #compute the probability tables between each one source and the target
    prob_yx1, prob_yx2, prob_yx3 = [np.zeros([10,2]) for i in xrange(3)]
    prob_yx1[:,0] = tripletprob[:,0]+tripletprob[:,2]+tripletprob[:,3]+tripletprob[:,6]
    prob_yx1[:,1] = tripletprob[:,1]+tripletprob[:,4]+tripletprob[:,5]+tripletprob[:,7]
    
    #print prob_yx1
    
    prob_yx2[:,0] = tripletprob[:,0]+tripletprob[:,1]+tripletprob[:,3]+tripletprob[:,5]
    prob_yx2[:,1] = tripletprob[:,2]+tripletprob[:,4]+tripletprob[:,6]+tripletprob[:,7]
    
    prob_yx3[:,0] = tripletprob[:,0]+tripletprob[:,1]+tripletprob[:,2]+tripletprob[:,4]
    prob_yx3[:,1] = tripletprob[:,3]+tripletprob[:,5]+tripletprob[:,6]+tripletprob[:,7]
    
    #print prob_yx3
    
    #compute the probability tables between two sources and the target
    prob_yx12, prob_yx13, prob_yx23 = [np.zeros([10,4]) for i in xrange(3)]
    #order is always 00, 10, 01, 11
    prob_yx12[:,0] = tripletprob[:,0]+tripletprob[:,3]
    prob_yx12[:,1] = tripletprob[:,1]+tripletprob[:,5]
    prob_yx12[:,2] = tripletprob[:,2]+tripletprob[:,6]
    prob_yx12[:,3] = tripletprob[:,4]+tripletprob[:,7]
    
    prob_yx13[:,0] = tripletprob[:,0]+tripletprob[:,2]
    prob_yx13[:,1] = tripletprob[:,1]+tripletprob[:,4]
    prob_yx13[:,2] = tripletprob[:,3]+tripletprob[:,6]
    prob_yx13[:,3] = tripletprob[:,5]+tripletprob[:,7]

    prob_yx23[:,0] = tripletprob[:,0]+tripletprob[:,1]
    prob_yx23[:,1] = tripletprob[:,2]+tripletprob[:,4]
    prob_yx23[:,2] = tripletprob[:,3]+tripletprob[:,5]
    prob_yx23[:,3] = tripletprob[:,6]+tripletprob[:,7]
    
    print '12'
    print prob_yx12
    print '13'
    print prob_yx13
    print '23'
    print prob_yx23
    
    #compute the appropriate mutual information terms
    MI1 = mutual_info(priors_y, priors_x1, prob_yx1)
    MI2 = mutual_info(priors_y, priors_x2, prob_yx2)
    MI3 = mutual_info(priors_y, priors_x3, prob_yx3)
    MI123 = mutual_info(priors_y, priors_x123, tripletprob)
    print MI123
        
    #compute the appropriate Imin terms
    R1_23 = redundancy_2_vars(prob_yx1,prob_yx23)
    R2_13 = redundancy_2_vars(prob_yx2,prob_yx13)
    R3_12 = redundancy_2_vars(prob_yx3,prob_yx12)
    R1_2_3 = redundancy_3_vars(prob_yx1,prob_yx2,prob_yx3)
    
    #compute the Imax term of sources (12),(13),(23)
    Imax12_12_23 = Imax_3_vars(prob_yx12,prob_yx13,prob_yx23)
    print Imax12_12_23
    
    #compute the unique information terms:
    U1 = MI1 - R1_23
    U2 = MI2 - R2_13
    U3 = MI3 - R3_12
    
    #compute the total synergy:
    S123 = MI123 - Imax12_12_23
    
    #compute the remaining PID terms all together:
    Remaining_terms = MI123 - S123 - R1_2_3 - U1 - U2 - U3
    
    #set terms to 0 if they are of round-off order of magnitude
    if U1 < 10**(-10):
        U1 = 0
    if U2 < 10**(-10):
        U2 = 0
    if U3 < 10**(-10):
        U3 = 0
    if S123 < 10**(-10):
        S123 = 0 
    if R1_2_3 < 10**(-10):
        R1_2_3 = 0
    if Remaining_terms < 10**(-10):
        Remaining_terms = 0
    
        
    return U1, U2, U3, R1_2_3, S123, Remaining_terms, MI1, MI2, MI3, MI123
    


def PID_2_return_probs(jointprob):
    '''
    PID2 using joint probabilities
    this only works for 2 sources since I'm picking out the appropriate values from
    the joint probability matrix which goes like:
    [[p(y=0,x1=0,x2=0), p(y=0,x1=1,x2=0), etc],[p(y=1,x1=0,x2=0), etc], etc]
    so first index is wrt the target variable
    '''
    
    #from the jointprob matrix, compute all the necessary things to give to the functions
    #print jointprob
    
    priors_y = np.sum(jointprob,1)
    priors_x12 = np.sum(jointprob,0)
    priors_x1 = np.array([priors_x12[0]+priors_x12[2], priors_x12[1]+priors_x12[3]])
    priors_x2 = np.array([priors_x12[0]+priors_x12[1], priors_x12[2]+priors_x12[3]])
    #this could maybe be shorter:
    
    joint_yx1, joint_yx2 = [np.zeros([jointprob.shape[0],2]), np.zeros([jointprob.shape[0],2])]
    joint_yx1[:,0] = jointprob[:,0] + jointprob[:,2]
    joint_yx1[:,1] = jointprob[:,1] + jointprob[:,3]
    joint_yx2[:,0] = jointprob[:,0] + jointprob[:,1]
    joint_yx2[:,1] = jointprob[:,2] + jointprob[:,3]
    
    #compute mutual info between {X1,X2} and Y        
    MI12 = mutual_info(priors_y, priors_x12, jointprob)
    #print 'MI12: %f' % MI12
    
    #compute mutual info between target and seperate sources
    MI1 = mutual_info(priors_y, priors_x1, joint_yx1)
    MI2 = mutual_info(priors_y, priors_x2, joint_yx2)  
    #print 'MI1: %f' % MI1
    
    #compute redundancy (i.e. shared info) between 2 sources
    #redundancy = red(priors_y, condprob_x1y, condprob_yx1, condprob_x2y, condprob_yx2)
    redundancy = redundancy_2(priors_y, priors_x1, priors_x2, joint_yx1, joint_yx2)
    
    #compute unique info by deducting redundancy from indiv mutual info terms    
    unique_x1 = unique(MI1,redundancy)
    unique_x2 = unique(MI2,redundancy)
    
    #compute synergy by deducting all other PID terms from mutual info between
    #both sources together and target
    syn = synergy(MI12, redundancy, unique_x1, unique_x2)
    
    return redundancy, unique_x1, unique_x2, syn, MI1, MI2, MI12, joint_yx1, joint_yx2, priors_x1, priors_x2

def PID3_terms_return_probs(tripletprob):
    '''
    This function computes for a given prob function of 3 binary sources and the target
    The following PID2 terms: U1, U2, U3, R123, S123.
    And outputs the remaining terms as 'remaining'
    
    Input is a 10 by 8 prob table where rows = target &
    columns = sources with the order 000,100,010,001,110,101,011,111
    
    
    Output:
    
    '''
    
    #compute the marginal probabilities by appropriate summing
    priors_y = np.sum(tripletprob,1)
    priors_x123 = np.sum(tripletprob,0)
    priors_x1 = np.array([priors_x123[0]+priors_x123[2]+priors_x123[3]+priors_x123[6], priors_x123[1]+priors_x123[4]+priors_x123[5]+priors_x123[7]])
    priors_x2 = np.array([priors_x123[0]+priors_x123[1]+priors_x123[3]+priors_x123[5], priors_x123[2]+priors_x123[4]+priors_x123[6]+priors_x123[7]])
    priors_x3 = np.array([priors_x123[0]+priors_x123[1]+priors_x123[2]+priors_x123[4], priors_x123[3]+priors_x123[5]+priors_x123[6]+priors_x123[7]])
    
    #compute the probability tables between each one source and the target
    prob_yx1, prob_yx2, prob_yx3 = [np.zeros([10,2]) for i in xrange(3)]
    prob_yx1[:,0] = tripletprob[:,0]+tripletprob[:,2]+tripletprob[:,3]+tripletprob[:,6]
    prob_yx1[:,1] = tripletprob[:,1]+tripletprob[:,4]+tripletprob[:,5]+tripletprob[:,7]
    
    #print prob_yx1
    
    prob_yx2[:,0] = tripletprob[:,0]+tripletprob[:,1]+tripletprob[:,3]+tripletprob[:,5]
    prob_yx2[:,1] = tripletprob[:,2]+tripletprob[:,4]+tripletprob[:,6]+tripletprob[:,7]
    
    prob_yx3[:,0] = tripletprob[:,0]+tripletprob[:,1]+tripletprob[:,2]+tripletprob[:,4]
    prob_yx3[:,1] = tripletprob[:,3]+tripletprob[:,5]+tripletprob[:,6]+tripletprob[:,7]
    
    #print prob_yx3
    
    #compute the probability tables between two sources and the target
    prob_yx12, prob_yx13, prob_yx23 = [np.zeros([10,4]) for i in xrange(3)]
    #order is always 00, 10, 01, 11
    prob_yx12[:,0] = tripletprob[:,0]+tripletprob[:,3]
    prob_yx12[:,1] = tripletprob[:,1]+tripletprob[:,5]
    prob_yx12[:,2] = tripletprob[:,2]+tripletprob[:,6]
    prob_yx12[:,3] = tripletprob[:,4]+tripletprob[:,7]
    
    prob_yx13[:,0] = tripletprob[:,0]+tripletprob[:,2]
    prob_yx13[:,1] = tripletprob[:,1]+tripletprob[:,4]
    prob_yx13[:,2] = tripletprob[:,3]+tripletprob[:,6]
    prob_yx13[:,3] = tripletprob[:,5]+tripletprob[:,7]

    prob_yx23[:,0] = tripletprob[:,0]+tripletprob[:,1]
    prob_yx23[:,1] = tripletprob[:,2]+tripletprob[:,4]
    prob_yx23[:,2] = tripletprob[:,3]+tripletprob[:,5]
    prob_yx23[:,3] = tripletprob[:,6]+tripletprob[:,7]
    
    #compute the appropriate mutual information terms
    MI1 = mutual_info(priors_y, priors_x1, prob_yx1)
    MI2 = mutual_info(priors_y, priors_x2, prob_yx2)
    MI3 = mutual_info(priors_y, priors_x3, prob_yx3)
    MI123 = mutual_info(priors_y, priors_x123, tripletprob)
    
    #compute the appropriate Imin terms
    R1_23 = redundancy_2_vars(prob_yx1,prob_yx23)
    R2_13 = redundancy_2_vars(prob_yx2,prob_yx13)
    R3_12 = redundancy_2_vars(prob_yx3,prob_yx12)
    R1_2_3 = redundancy_3_vars(prob_yx1,prob_yx2,prob_yx3)
    
    #compute the Imax term of sources (12),(13),(23)
    Imax12_13_23 = Imax_3_vars(prob_yx12,prob_yx13,prob_yx23)
    
    #compute the unique information terms:
    U1 = MI1 - R1_23
    U2 = MI2 - R2_13
    U3 = MI3 - R3_12
    
    #compute the total synergy:
    S123 = MI123 - Imax12_13_23
    
    #compute the remaining PID terms all together:
    Remaining_terms = MI123 - S123 - R1_2_3 - U1 - U2 - U3
    
    #set terms to 0 if they are of round-off order of magnitude
    if U1 < 10**(-10):
        U1 = 0
    if U2 < 10**(-10):
        U2 = 0
    if U3 < 10**(-10):
        U3 = 0
    if S123 < 10**(-10):
        S123 = 0 
    if R1_2_3 < 10**(-10):
        R1_2_3 = 0
    if Remaining_terms < 10**(-10):
        Remaining_terms = 0
    
        
    return U1, U2, U3, R1_2_3, S123, Remaining_terms, MI1, MI2, MI3, MI123, prob_yx1, prob_yx2, prob_yx3, priors_x1, priors_x2, priors_x3
    
#for now I decided to adapt all PID terms using surrogate testing, so this is not necessary yet.
#shuffled_jointprob(labels, hidden_states, neuron1, neuron2, shuffle_number)
#    '''
#    Function used for surrogate testing. given the necessary input to compute 
#    the joint probability table between sources and target it suffles
#    
#    '''
    

"""
#To test for 'AND'-GATE:
jointprob = np.array([[0.25,0.25,0.25,0],[0,0,0,0.25]])
"""


def compute_jointprob_full(labels, hidden_states, selection):
    '''
    Given an array with numbers corresponding to the selected hidden neurons
    this function will compute the joint probability table of this selection
    of neurons and the target variable.

    The 'selection' array is first sorted and then the probability table is filled
    by counting, and keeping order according to the decimal representation of
    the binary array.
    '''
    if labels.shape[0] != hidden_states.shape[0]:
        raise TypeError('each data point needs to have hidden states associated!!')

    selection = np.sort(selection)
def
    hidden_statesOI = np.copy(hidden_states)
    hidden_statesOI = hidden_statesOI[:, selection]

    jointprob = np.zeros([10, 2 ** len(selection)])

    for i in xrange(labels.shape[0]):
        vect = hidden_statesOI[i, :]
        index = intefy(vect)
        label = np.where(labels[i, :] == 1)[0]
        jointprob[label, index] += 1.0

    jointprob = jointprob * 1.0 / labels.shape[0]

    return jointprob

def intefy(x):
    '''
    Makes binary vector into integer
    '''
    return sum(1<<i for i, b in enumerate(x) if b)


