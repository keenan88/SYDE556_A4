
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np

#nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

model = nengo.Network()

def y(x):
    return 1 - 2*x

with model:
    
    with nengo.Network() as net:
        
        N1 = 100
        N2 = 50
        
        radius_ = 1
            
        stim = nengo.Node(lambda t: 0.0 if t < 0.1 else (1.0 if t < 0.4 else 0.0))
        #stim = nengo.Node(lambda t: t)
        """
        stim = nengo.Node(nengo.processes.Piecewise({
            0.1: 1.0,
            0.4: 0.0
            })
        )
        """
        
        # Noise totally murks this
        ens1 = nengo.Ensemble(
            radius = radius_,
            n_neurons = N1, 
            dimensions = 1,
            encoders = np.matrix(np.random.choice([-1, 1], size = N1)).T,
            max_rates = np.random.uniform(low = 100, high = 200, size=N1),
            #intercepts = np.random.uniform(low = -1, high = 1, size=N),
            #noise = nengo.processes.WhiteNoise(dist = nengo.dists.Gaussian(mean=0, std=0.1 * 200)),
            neuron_type = nengo.neurons.LIFRate(tau_rc = 0.02, tau_ref = 0.002)
        )
        
        ens2 = nengo.Ensemble(
            radius = radius_,
            n_neurons = N2, 
            dimensions = 1,
            encoders = np.matrix(np.random.choice([-1, 1], size = N2)).T,
            max_rates = np.random.uniform(low = 100, high = 200, size=N2),
            #intercepts = np.random.uniform(low = -1, high = 1, size=N),
            #noise = nengo.processes.WhiteNoise(dist = nengo.dists.Gaussian(mean=0, std=0.1 * 200)),
            neuron_type = nengo.neurons.LIFRate(tau_rc = 0.02, tau_ref = 0.002)
        )
        
        stim_connection = nengo.Connection(
            stim, 
            ens1,
            synapse = nengo.synapses.Alpha(tau = 0.01)
        )
    
        feed_forward = nengo.Connection(
            ens1, 
            ens2,
            synapse = nengo.synapses.Alpha(tau = 0.01),
            function = y
        )
        
        
        probe_stim = nengo.Probe(stim)
        probe_conn1 = nengo.Probe(stim_connection)
        probe_conn2 = nengo.Probe(feed_forward)
        probe_ens1 = nengo.Probe(ens1)
        probe_ens2 = nengo.Probe(ens2)
        
        
        
        with nengo.Simulator(net, progress_bar=False) as sim:
            
            
            sim.run(0.5)
            
            plt.plot(sim.trange(), sim.data[probe_stim], label='Stimulus')
            #plt.plot(sim.trange(), sim.data[probe_conn1], label='Connection 1')
            plt.plot(sim.trange(), sim.data[probe_ens1], label='Ensemble 1')
            plt.plot(sim.trange(), sim.data[probe_ens2], label='Ensemble 2')
            #plt.plot(sim.trange(), sim.data[probe_conn2], label='Connection 2')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Neuron Output')
            plt.show()
            
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    