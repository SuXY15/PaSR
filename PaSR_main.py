# -*- coding: utf-8 -*-
from utils import *
from PaSR_models import *

# ====================
# PaSR simulation, particle weights are not used
def run_simulation(mech, case, T0, P, eq_ratio, fuel, oxidizer, mix_model="IEM",
                   Np=100,  tau_res=(10./1000.), tau_mix=(1./1000.), num_res=10,
                   sigma_k=None,  doplot=False):
    """Perform partially stirred reactor (PaSR) simulation.
    Parameters
    ----------
    mech : str
        Mechanism filename (in Cantera format).
    case : {'Premixed','Non-premixed'}
        Case of PaSR simulation; {'Premixed', 'Non-premixed'}.
    T0 : float
        Initial temperature [K].
    P : float
        Pressure [atm].
    eq_ratio : float
        Equivalence ratio.
    fuel : dict
        Dictionary of molecules in the fuel mixture and the fraction of \
        each molecule in the fuel mixture.
    oxidizer : dict
        Dictionary of molecules in the oxidizer mixture and the \
        fraction of each molecule in the oxidizer mixture.
    Np : Optional[int]
        Number of particles. Optional, default 100.
    tau_res : Optional[float]
        Residence time [s]. Optional, default 10 [ms].
    tau_mix : Optional[float]
        Mixing timescale [s]. Optional, default 1 [ms].
    num_res : Optional[int]
        Numer of residence times to simulate. Optional, default 5.
    Returns
    -------
    particle_data : numpy.array
        numpy.array with full particle data.
    """
    # Time step control
    dt = 0.1 * min(tau_res, tau_mix)

    if mix_model=="EMST" or mix_model=="EMST1D": # EMST need smaller timestep
        dt /= 5

    time_end = num_res * tau_res
    num_steps = int(time_end / dt + 10)

    # Set initial conditions
    gas = ct.Solution(mech)
    gas.TP = T0, P * ct.one_atm

    # get Zst and Zeq
    gas.set_equivalence_ratio(1.0, fuel, oxidizer)
    Zst = gas.mixture_fraction(fuel, oxidizer)
    gas.set_equivalence_ratio(eq_ratio, fuel, oxidizer)
    Zeq = gas.mixture_fraction(fuel, oxidizer)

    # Inlet streams
    print("\tSetting inlet streams ...")
    if case.lower() == 'premixed':
        # Premixed
        flow_rates = dict(fuel_oxid = 0.95, pilot = 0.05)
    elif case.lower() == 'non-premixed':
        # Non-premixed
        flow_rates = dict(fuel = Zeq, oxid = 1-Zeq, pilot = 0.0)
    else:
        print('Error: case needs to be either premixed or non-premixed.')
        sys.exit(1)

    inlet_streams = []
    for src in flow_rates.keys():
        if src == 'fuel':
            fuel_gas = ct.Solution(mech)
            fuel_gas.TPX = T0, P * ct.one_atm, fuel
            fuel_stream = Stream(fuel_gas, flow_rates['fuel'])
            inlet_streams.append(fuel_stream)
        elif src == 'oxid':
            oxid_gas = ct.Solution(mech)
            oxid_gas.TPX = T0, P * ct.one_atm, oxidizer
            oxid_stream = Stream(oxid_gas, flow_rates['oxid'])
            inlet_streams.append(oxid_stream)
        elif src == 'pilot':
            pilot_gas = ct.Solution(mech)
            pilot_gas.TP = T0, P * ct.one_atm
            pilot_gas.set_equivalence_ratio(eq_ratio, fuel, oxidizer)
            pilot_gas.equilibrate('HP')
            pilot_stream = Stream(pilot_gas, flow_rates['pilot'])
            inlet_streams.append(pilot_stream)
        elif src == 'fuel_oxid':
            fuel_oxid_gas = ct.Solution(mech)
            fuel_oxid_gas.TP = T0, P * ct.one_atm,
            fuel_oxid_gas.set_equivalence_ratio(eq_ratio, fuel, oxidizer)
            fuel_oxid_stream = Stream(fuel_oxid_gas, flow_rates['fuel_oxid'])
            inlet_streams.append(fuel_oxid_stream)

    # Initialize, %60 as Zst, %40 random Z
    print("\tInitializing particles ...")
    particles = []
    Nst = int(Np*0.6)
    for i in range(Nst):
        gas = ct.Solution(mech)
        gas.TP = T0, P * ct.one_atm
        gas.set_equivalence_ratio(1.0, fuel, oxidizer)
        gas.equilibrate('HP')
        particles.append(Particle(gas))
    for i in range(Np-Nst):
        Z = i/(Np-Nst)
        gas = ct.Solution(mech)
        gas.TP = T0, P * ct.one_atm
        gas.set_equivalence_ratio(Z/(1-Z)*(1-Zst)/Zst, fuel, oxidizer)
        gas.equilibrate('HP')
        particles.append(Particle(gas))

    # prepare for loop
    t = 0.0
    i_step = 0
    part_out = 0.0
    times = np.zeros(num_steps + 1)
    temp_mean = np.zeros(num_steps + 1)
    temp_mean[i_step] = np.mean([p.gas.T for p in particles])

    # Array of full particle data for all timesteps
    particle_data = np.empty([num_steps + 1, Np, gas.n_species + 3])
    save_data(i_step, t, particles, particle_data)

    print('Time [ms]  Temperature [K]')
    print('t/tres = {:4.2f},  <T> = {:6.1f}'.format(t/tau_res, temp_mean[i_step]))

    while t < time_end:
        t0 = time.time()

        if (t + dt) > time_end:
            dt = time_end - t
        else:
            dt = dt

        # ==========
        # flow in/out
        part_out += Np * dt / tau_res
        npart_out = int(round(part_out))
        part_out -= npart_out

        idxs = np.random.choice(Np, npart_out, replace=False)
        for i in idxs:
            i_str = inflow(inlet_streams)
            particles[i](inlet_streams[i_str]())

        t1 = time.time()

        # ==========
        # mixing
        mix_substep(particles, dt, tau_mix,
                    fuel, oxidizer, model=mix_model, sigma_k=sigma_k)

        t2 = time.time()

        # ==========
        # reacting
        reaction_substep(particles, dt, mech)

        t3 = time.time()

        # ==========
        # Save states
        t += dt
        i_step += 1

        temp_mean[i_step] = np.mean([p.gas.T for p in particles])
        times[i_step] = t
        save_data(i_step, t, particles, particle_data)

        print('t/tres = {:4.2f},  <T> = {:6.1f}'.format(t/tau_res, temp_mean[i_step]))
        # print('cost  of flow {:.1e}, mix {:.1e}, react {:.1e}'.format(
        #                       t1-t0,      t2-t1,        t3-t2)

        # ==========
        # plot
        if doplot:
            plt.ion()
            plt.figure(9, figsize=(5,4))
            plt.cla()
            T_arr = [p.gas.T for p in particles]
            Z_arr = [p.gas.mixture_fraction(fuel, oxidizer) for p in particles]
            plt.plot(Z_arr, T_arr, 'k.', ms=0.8, alpha=0.8)
            plt.xlim([0,1])
            plt.ylim([0, 2500])
            plt.xlabel("Z")
            plt.ylabel("T")
            plt.draw()
            plt.pause(1e-9)
            
    times = times[:i_step + 1]
    temp_mean = temp_mean[:i_step + 1]
    particle_data = particle_data[:i_step + 1, :, :]

    return particle_data


if __name__ == "__main__":
    # =================================
    # parse input args
    print("Parsing input arguments ...")
    parser = ArgumentParser(description='Runs partially stirred reactor '
                                        '(PaSR) simulation.')
    parser.add_argument('-i', '--input',
                        type=str, required=True,
                        help='Input file in YAML format for PaSR simulation.')
    parser.add_argument('-o', '--output',
                        type=str, default='none',
                        help='PaSR results file (.npy).')
    parser.add_argument('-p', '--doplot', action='store_true',
                        help='Do plot or not')
    args = parser.parse_args()
    
    if args.output=="none":
        casename = os.path.split(args.input)[-1].split(".")[0]
        args.output = "data/" + casename + ".npy"
    
    inputs = parse_input_file(args.input)

    # =================================
    # compute and save
    if not checkexists(args.output):
        print("Computing PaSR ...")
        particle_data = run_simulation(
            inputs['mech'], inputs['case'], inputs['temperature'], inputs['pressure'],
            inputs['equivalence ratio'], inputs['fuel'], inputs['oxidizer'],
            inputs['mixing model'], inputs['number of particles'],
            inputs['residence time'], inputs['mixing time'], 
            inputs['number of residence times'], inputs['sigma_k'], args.doplot)
        np.save(args.output, particle_data)

    # =================================
    # show results
    print("Loading PaSR data", args.output)
    gas = ct.Solution(inputs['mech'])
    particle_data = np.load(args.output)
    Ntimes = particle_data.shape[0]
    Nparticles = particle_data.shape[1]

    # get Zst
    gas.TP = inputs['temperature'], inputs['pressure']*ct.one_atm
    gas.set_equivalence_ratio(1.0, inputs['fuel'], inputs['oxidizer'])
    Zst = gas.mixture_fraction(inputs['fuel'], inputs['oxidizer'])
    
    # get equilibrium line
    mfs = np.linspace(0,1,100)
    Teq = []
    for Z in mfs:
        gas.TP = inputs['temperature'], inputs['pressure']*ct.one_atm
        if Z!=1:
            gas.set_equivalence_ratio(Z/(1-Z)*(1-Zst)/Zst, inputs['fuel'], inputs['oxidizer'])
        else:
            gas.X = inputs['fuel']
        gas.equilibrate('HP')
        Teq.append(deepcopy(gas.T))

    # figure 1: temperature
    plt.figure(1, figsize=(5,4))
    t_arr = particle_data[:,0,0]
    T_mean = np.mean(particle_data[:,:,1], axis=1)
    T_std = np.std(particle_data[:,:,1], axis=1)
    idx = range(0, Ntimes-1, 10)
    plt.errorbar(t_arr[idx], T_mean[idx], yerr=T_std[idx], label=r"$T_{mean} \pm T_{std}$")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\langle Y \rangle \pm Y_{rms}$")

    # figure 2: Z-T plot
    plt.figure(2, figsize=(5,4))
    plt.plot(mfs, Teq, 'r-', lw=0.5)
    mf_arr = []
    for j in range(Ntimes-100, Ntimes, 5):
        mf = np.array([get_mixture_fraction(gas, inputs['fuel'], inputs['oxidizer'], 
                particle_data[j,i,3:]) for i in range(Nparticles)])
        Tp = particle_data[j,:,1]
        mf_arr.append(mf)
        plt.plot(mf, Tp, 'k.', ms=0.8, alpha=0.2)
    plt.xlim([0,1])
    plt.ylim([0,2500])
    plt.title(casename)
    plt.xlabel("Mixture Fraction")
    plt.ylabel("Temperature [K]")
    plt.subplots_adjust(left=0.16, right=0.95, top=0.9, bottom=0.15)
    plt.savefig("figs/"+casename+"_Z-T.png")

    # figure 3: Z PDF
    Z = np.array(mf_arr).flatten()
    plt.figure(3, figsize=(5,4))
    PDF,BIN = np.histogram(Z, bins=20, density=True)
    BIN = (BIN[1:] + BIN[:-1])/2
    plt.plot(BIN, PDF)
    plt.xlim([0,1])
    plt.xlabel("Z")
    plt.ylabel("PDF")

    # plt.figure(4, figsize=(5,4))
    # for j in range(Ntimes-1, Ntimes, 1):
    #     mf = np.array([get_mixture_fraction(gas, inputs['fuel'], inputs['oxidizer'], 
    #             particle_data[j,i,3:]) for i in range(Nparticles)])
    #     Tp = particle_data[j,:,1+gas.species_index("OH")]
    #     plt.plot(mf, Tp, 'k.', ms=0.8, alpha=0.8)
    # plt.xlim([0,1])
    # plt.title(casename)
    # plt.xlabel("Mixture Fraction")
    # plt.ylabel("Y_{OH}")
    # plt.subplots_adjust(left=0.16, right=0.95, top=0.9, bottom=0.15)

    plt.show()
