
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np


model = nengo.Network()


with model:
    
    with nengo.Network() as net:
        
        N1 = 200
        
        radius_ = 1
            
        stim = nengo.Node(lambda t: 5 * np.sin(5*t))
        
        # Noise totally murks this
        ens1 = nengo.Ensemble(
            radius = radius_,
            n_neurons = N1, 
            dimensions = 1,
            encoders = np.matrix(np.random.choice([-1, 1], size = N1)).T,
            max_rates = np.random.uniform(low = 100, high = 200, size=N1),
            #intercepts = np.random.uniform(low = -1, high = 1, size=N),
            #noise = nengo.processes.WhiteNoise(dist = nengo.dists.Gaussian(mean=0, std=0.1 * 200)),
            neuron_type = nengo.neurons.SpikingRectifiedLinear()
        )
        
        stim_connection = nengo.Connection(
            stim, 
            ens1,
            synapse = nengo.synapses.Alpha(tau = 0.005)
        )
    
        loopback = nengo.Connection(
            ens1, 
            ens1,
            synapse = nengo.synapses.Alpha(tau = 0.05),
        )
        
        
        probe_stim = nengo.Probe(
            target = stim, 
            synapse = nengo.synapses.Alpha(tau = 0.01)
        ) 
        
        probe_conn1 = nengo.Probe(
            target = stim_connection,
            synapse = nengo.synapses.Alpha(tau = 0.01)
        )
        
        probe_ens1 = nengo.Probe(
            ens1,
            synapse = nengo.synapses.Alpha(tau = 0.01)
        )
        
        
        
        with nengo.Simulator(net, progress_bar=False) as sim:
            
            sim.run(1.5)
            
            time = np.linspace(0, 1.5, 1000)
            dt = (1.5 - 0) / 1000
            f_sum = 0
            f_des = []
            for i in range(len(time)):
                t = time[i]
                f = 5 * np.sin(5*t)
                f_sum += f * dt
                f_des.append(f_sum)
                
                
            
            plt.plot(sim.trange(), sim.data[probe_stim], label='Stimulus')
            plt.plot(sim.trange(), sim.data[probe_conn1], label='Ensemble Input')
            plt.plot(sim.trange(), sim.data[probe_ens1], label='Ensemble Output')
            plt.plot(time, f_des, label='Desired output')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Neuron Output')
            plt.show()
            
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    