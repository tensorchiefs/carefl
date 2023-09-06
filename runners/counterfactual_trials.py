# trial counterfactual predictions of flow models
#
#

import numpy as np
import os
import pylab as plt
import seaborn as sns

from data.generate_synth_data import intervention_sem
from models.carefl import CAREFL


def counterfactuals(args, config):
    # we generate the same SEM as in the intervention example

    X_org, _, _ = intervention_sem(n_obs=2500, seed=0, random=False)
    (_, _, X3_std, X4_std) = X_org.std(axis=0)
    X = X_org / np.array([1, 1, X3_std, X4_std])
    # fit CAReFl to the data
    mod = CAREFL(config)
    mod.fit_to_sem(X)

    def gen_observation(N):
        X_0 = N[0, 0]
        X_1 = N[0, 1]
        X_2 = (X_0 + .5 * X_1 * X_1 * X_1 + N[0, 2]) / X3_std
        X_3 = (-X_1 + .5 * X_0 * X_0 + N[0, 3]) / X4_std
        return np.array([X_0, X_1, X_2, X_3]).reshape((1, 4))

    ### now we run some CF trials:
    N = np.array([2, 1.5, 1.4, -1]).reshape((1, 4))
    # this is the random value of 4D random varibale x we observe. Now we plot the counterfactual
    # given x_0 had been other values
    xObs = gen_observation(N)  # should be (2.00, 1.50, 0.81, −0.28)
    #in the paper (2.00, 1.50, 0.81, −0.28)
    #in realiy array([[ 2.        ,  1.5       ,  0.84645028, -0.26158623]])
    ### Hack Attack ###
    print("An numpy array of observational data, has been created")
    print(X)
    print(X.mean(axis=0))
    print(X.std(axis=0))
    #writes the data to a csv file
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/X.csv", X, delimiter=",")
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/X_org.csv", X_org, delimiter=",")
    print(xObs)
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/xObs.csv", xObs, delimiter=",")
    
    #Is consitent same as xObs
    mod.predict_counterfactual(x_obs=xObs, cf_val=xObs[0,0], iidx=0)  
    gen_observation(xObs)
    
    
    #### End of Hack Attack ####

    # PLOT
    xvals = np.arange(-3, 3, .1)
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # fig.suptitle(r'Counterfactual predictions', fontsize=22)
    # see quality of counterfactual predictions for X_4
    xCF_true = [gen_observation(np.hstack((x, N[0, 1:])).reshape((1, 4)))[0, 3] for x in xvals]
    xCF_pred = [mod.predict_counterfactual(x_obs=xObs, cf_val=x, iidx=0)[0, 3] for x in xvals]
    ### Hack Attack do on X1 ###
    print("xCF_true: ", xCF_true)
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/xCF_onX1_true.csv", xCF_true, delimiter=",")
    print("xCF_pred: ", xCF_pred)
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/xCF_onX1_pred.csv", xCF_pred, delimiter=",")
    ### Hack Attack do on X1 ###

    ax1.plot(xvals, xCF_true, label=r'True $\mathbb{E} [{X_4}_{X_1 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax1.plot(xvals, xCF_pred, label=r'Predicted $\mathbb{E} [{X_4}_{X_1 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax1.legend(loc=1, fontsize=15)
    ax1.set_xlabel(r'Value of counterfactual variable, $X_1=\alpha$', fontsize=18)
    ax1.set_ylabel(r'Predicted value of $X_4$', fontsize=18)
    # see quality of counterfactual predictions for X_3
    xCF_true = [gen_observation(np.hstack((N[0, 0], x, N[0, 2:])).reshape((1, 4)))[0, 2] for x in xvals]
    xCF_pred = [mod.predict_counterfactual(x_obs=xObs, cf_val=x, iidx=1)[0, 2] for x in xvals]
     ### Hack Attack do on X1 ###
    print("xCF_true: ", xCF_true)
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/xCF_onX2_true.csv", xCF_true, delimiter=",")
    print("xCF_pred: ", xCF_pred)
    np.savetxt("/Users/oli/Documents/GitHub/causality/data/CAREFL_CF/xCF_onX2_pred.csv", xCF_pred, delimiter=",")
    ### Hack Attack do on X1 ###
    
    ax2.plot(xvals, xCF_true, label=r'True $\mathbb{E} [{X_3}_{X_2 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax2.plot(xvals, xCF_pred, label=r'Predicted $\mathbb{E} [{X_3}_{X_2 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax2.legend(loc='best', fontsize=15)
    ax2.set_xlabel(r'Value of counterfactual variable, $X_2=\alpha$', fontsize=18)
    ax2.set_ylabel(r'Predicted value of $X_3$', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.925)
    plt.savefig(os.path.join(args.run, 'counterfactuals_4d.pdf'), dpi=300)
