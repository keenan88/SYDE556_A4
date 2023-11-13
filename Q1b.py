
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np

#Resource: https://www.nengo.ai/nengo/examples/usage/tuning-curves.html

model = nengo.Network()


with model:
    N = 100
    
    rmses = []
    
    # How to seed the nengo network?
    for radius_ in [0.5, 1, 2, 4]:
    
        # LIF Neuron. TauRC and Tauref??
        ens_1d = nengo.Ensemble(
            radius = radius_,
            n_neurons = N, 
            dimensions = 1,
            encoders = np.matrix(np.random.choice([-1, 1], size = N)).T,
            max_rates = np.random.uniform(low = 100, high = 200, size=N),
            #intercepts = np.random.uniform(low = -1, high = 1, size=N),
            noise = nengo.processes.WhiteNoise(dist = nengo.dists.Gaussian(mean=0, std=0.1 * 200)),
            neuron_type = nengo.neurons.LIFRate(tau_rc = 0.02, tau_ref = 0.002)
        )
        
        loopback = nengo.Connection(ens_1d, ens_1d)
                
        eval_points, activities = \
        nengo.utils.ensemble.tuning_curves(ens_1d, nengo.Simulator(model))
        
        plt.figure()
        plt.plot(eval_points, activities)
        plt.ylabel("Firing rate (Hz)")
        plt.xlabel("Input scalar, x")
        plt.show()
    
        # How to add noise?
        eval_points = np.matrix(np.linspace(-1, 1, 100)).T
        _, targets, decoded = \
        nengo.utils.connection.eval_point_decoding(loopback, nengo.Simulator(model), eval_points)
        
        plt.figure()
        plt.plot(eval_points, targets, label="Targets")
        plt.plot(decoded, targets, label="Decoded")
        plt.legend()
        plt.ylabel("Firing rate (Hz)")
        plt.xlabel("Evaluation Points")
        
        rmse = np.sqrt(np.mean(np.square(targets - decoded)))
        rmses.append(rmse)
        
        print("Radius: ", radius_, "RMSE: ", rmse)
            
    
print(rmses)
    
    
    
    
    
    
    
    
    
    
    