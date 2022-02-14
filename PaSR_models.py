from utils import *

import yaml
from yaml import Loader
from argparse import ArgumentParser

# Parallel processing for reaction substep
parallel = True
try:
    import multiprocessing
    Nproc = multiprocessing.cpu_count() // 4
except ImportError:
    print('Warning: multiprocessing not installed')
    parallel = False

# For EMST, currently networkx is used
# it's possible to use github.com/wangyiqiu/pargeo for further acceleration
import mlpack
from networkx import Graph, minimum_spanning_edges
def emst_networkx(phis):
    g = Graph()
    for i,pi in enumerate(phis):
        for j,pj in enumerate(phis):
            if j<i:
                g.add_edge(i,j,weight=np.linalg.norm(pi-pj))
    return {'output':list(minimum_spanning_edges(g))}

class Stream(object):
    """Class for inlet flow stream into reactor.
    """
    def __init__(self, gas, flow):
        self.comp = np.hstack((gas.enthalpy_mass, gas.Y))
        self.flow = flow
        # Running variable of flow rate
        self.xflow = 0.0

    def __call__(self):
        return self.comp

class Particle(object):
    """Class for particle in reactor.
    """
    particle_mass = 0.1

    def __init__(self, gas):
        """Initialize particle object with thermochemical state.
        Parameters
        ----------
        gas : `cantera.Solution`
            Initial thermochemical state of particle
        Returns
        -------
        None
        """
        self.gas = gas

    def __call__(self, comp=None):
        """Return or set composition.
        Parameters
        ----------
        comp : Optional[cantera.Solution]
        Returns
        -------
        comp : numpy.array
            Thermochemical composition of particle (enthalpy + mass fractions).
        """
        if comp is not None:
            if isinstance(comp, Particle):
                h = comp.gas.enthalpy_mass
                Y = comp.gas.Y
            elif isinstance(comp, np.ndarray):
                h = comp[0]
                Y = comp[1:]
            else:
                return NotImplemented
            self.gas.HPY = h, self.gas.P, Y
        else:
            return np.hstack((self.gas.enthalpy_mass, self.gas.Y))

    def react(self, dt):
        """Perform reaction timestep by advancing network.
        Parameters
        ----------
        dt : float
            Reaction timestep [seconds]
        Returns
        -------
        None
        """
        reac = ct.IdealGasConstPressureReactor(self.gas,
            volume=Particle.particle_mass/self.gas.density)
        netw = ct.ReactorNet([reac])
        netw.advance(netw.time + dt)

def get_mixture_fraction(gas, fuel, oxidizer, Y):
    gas.Y = Y
    mf = gas.mixture_fraction(fuel, oxidizer)
    return mf

def pairwise(iterable):
    """Takes list of objects and converts into list of pairs.
    s -> (s0,s1), (s2,s3), (s4, s5), ...
    Parameters
    ----------
    iterable : list
        List of objects.
    Returns
    -------
    zipped : zip
        Zip with pairs of objects from `iterable`.
    """
    a = iter(iterable)
    return zip(a, a)

def mix_substep(particles, dt, tau_mix, fuel, oxidizer, model="IEM", sigma_k=None):
    """Pairwise mixing step.
    Parameters
    ----------
    particles : list of `Particle`
        List of `Particle` objects.
    dt : float
        Time step [s] to increment particles.
    tau_mix : float
        Mixing timescale [s].
    Returns
    -------
    None
    """

    omdt = 1./tau_mix * dt
    Np = len(particles)
    phis = np.array([p() for p in particles])

    # ==================
    # IEM Mixing
    if model=="IEM":
        phi_avr = np.mean(phis, axis=0)
        for i,phi in enumerate(phis):
            phis[i] += - 1/2. * omdt * (phi - phi_avr)

    # ==================
    # MC Mixing, particle weights are not used
    elif model=="MC":
        nmix = int(1.5 * omdt * Np + 1)
        pmix = 1.5 * omdt * Np / nmix

        for i in range(nmix):
            p = int(np.floor(np.random.rand()*Np))
            q = int(np.floor(np.random.rand()*Np))
            if np.random.rand() < pmix:
                a = np.random.rand()
                phi_pq = (phis[p] + phis[q])/2
                phis[p] += -a*(phis[p] - phi_pq)
                phis[q] += -a*(phis[q] - phi_pq)

    # ==================
    # EMST Mixing, 1-D version in Z-space, without aging strategy
    elif model=="EMST1D":
        # get mixture fractions
        Zs = np.array([p.gas.mixture_fraction(fuel,oxidizer) for p in particles])
        Ms = np.array([p.particle_mass for p in particles])

        dt_in = deepcopy(dt)
        while dt_in > 0:
            varPhi = np.var(phis, axis=0)
            sorted_id = np.argsort(Zs)

            w = Ms[sorted_id] / np.sum(Ms)
            W = np.cumsum(w)[:-1]
            Wv = np.array([min(Wi,1-Wi) for Wi in W])
            B = 2*Wv

            dphi = np.zeros_like(phis)
            for v in range(Np-1):
                mv = sorted_id[v]
                nv = sorted_id[v+1]
                dphi[mv] += - B[v] * (phis[mv] - phis[nv]) / w[mv]
                dphi[nv] += - B[v] * (phis[nv] - phis[mv]) / w[nv]
            
            AA = np.mean(dphi**2, axis=0)
            BB = 2*np.mean(dphi * phis, axis=0)
            CC = 1./tau_mix * varPhi
            dt = min(dt, np.min(1. * BB**2 / (4*AA*CC)))
            alphaPhi = ( -BB + np.sqrt(abs(BB*BB-4*AA*CC*dt)) ) / (2*AA*dt)
            alphaPhi = np.min(alphaPhi)
            
            dt = dt_in if dt_in <= dt else dt

            # root finding process
            for i in range(4):
                varNew = np.var(phis+dphi*alphaPhi*dt, axis=0)
                varDecay = 1 - np.mean(varNew / varPhi)
                varRatio = varDecay / (1-np.exp(-dt/tau_mix))
                alphaPhi = alphaPhi/varRatio
                print("   ", varDecay, varRatio)

            for i in range(Np):
                phis[i] += dphi[i] * alphaPhi * dt

            dt_in -= dt

            print("[DEBUG] EMST looping dt_in = %6.1e, dt = %6.1e, alpha = %6.1e"%(dt_in, dt, alphaPhi))
            # sys.exit(0)

    # ==================
    # EMST Mixing, (Ns+1)-D version in composition space, without aging strategy
    elif model=="EMST":
        Ms = np.array([p.particle_mass for p in particles])
        dt_in = deepcopy(dt)       

        scales = np.max(phis, axis=0) - np.min(phis, axis=0)
        scales[scales<1e-8] = 1e-8

        while dt_in > 0:
            # varPhi = np.var(phis, axis=0)
            varPhi = np.diag(np.cov(phis, rowvar=False))

            # generate EMST tree, 
            # edges = mlpack.emst(phis/scales)['output']   # using mlpack, unstable
            edges = emst_networkx(phis/scales)['output'] # using networkx, slower
            edges = [[int(e[0]),int(e[1])] for e in edges]
            
            # get nodes' children from edge list
            tree = {}
            for [mv, nv] in edges:
                if mv not in tree.keys(): tree[mv] = []
                if nv not in tree.keys(): tree[nv] = []
                tree[mv].append(nv)
                tree[nv].append(mv)
            
            # DFS for subtree weights
            # (below are nodes-edges of 10 particles)
            #    0  5  7 -- 6
            #     \ | /
            #  1 -- 2 -- 4
            #     /   \ 
            #    3     8 -- 9
            w = Ms / np.sum(Ms) # node weights
            WT = {}              # subtree weights:
            for i in range(Np):         # e.g., for the edge connecting nodes 2 and 8
                WT[i] = {}              #   W[2][8] = weight(2->8) = sum(w0,...,w7)
            visited = np.zeros(Np)      #   W[8][2] = weight(8->2) = sum(w8,w9)
            checked = np.zeros(Np)
            stack = [0]
            while len(stack) > 0:
                node = stack.pop()
                childs = tree[node]
                visited[node] = 1
                for child in childs:
                    if not visited[child]:
                        stack.append(node)
                        stack.append(child)
                        break
                # if only one child/neighbor is not checked
                # `node` can be checked with weights
                if np.sum(1-checked[childs]) == 1:
                    Wi = w[node]
                    for child in childs:
                        if checked[child]:
                            Wi += WT[child][node]
                    for child in childs:
                        if not checked[child]:
                            WT[node][child] = Wi
                            WT[child][node] = 1-Wi
                    checked[node] = 1

            # get edge-weight from subtree weights
            W = np.zeros(len(edges))
            for v,[mv,nv] in enumerate(edges):
                W[v] = min(WT[mv][nv], WT[nv][mv])
            
            # get edge-coefficient
            B = 2*W

            # get dphi/dt
            dphi = np.zeros_like(phis)
            for v,[mv,nv] in enumerate(edges):
                dphi[mv] += - B[v] * (phis[mv] - phis[nv]) / w[mv]
                dphi[nv] += - B[v] * (phis[nv] - phis[mv]) / w[nv]

            AA = np.mean(dphi**2, axis=0)
            BB = 2*np.mean(dphi * phis, axis=0)
            CC = 1./tau_mix * varPhi
            dt = min(dt, np.min(1. * BB**2 / (4*AA*CC)))
            alphaPhi = ( -BB + np.sqrt(abs(BB*BB-4*AA*CC*dt)) ) / (2*AA*dt)
            alphaPhi = np.min(alphaPhi)

            dt = dt_in if dt_in <= dt else dt

            # root finding process
            for i in range(4):
                varNew = np.diag(np.cov(phis+dphi*alphaPhi*dt, rowvar=False))
                varDecay = 1-np.mean(varNew / varPhi)
                varRatio = varDecay / (1-np.exp(-dt/tau_mix))
                alphaPhi = alphaPhi / varRatio
                # print("   ", varDecay, varRatio)

            for i in range(Np):
                phis[i] += dphi[i] * alphaPhi * dt

            dt_in -= dt

            # print("[DEBUG] EMST looping dt_in = %6.1e, dt = %6.1e, alpha = %6.1e"%(dt_in, dt, alphaPhi))
            # sys.exit(0)

    elif model=="KerM":
        if sigma_k is None:
            print("ERROR: KerM mixing model need sigma_k")
            sys.exit(1)
        if sigma_k < 0.01:
            print("Warning: sigma_k is recommended to be in [0.01 ~ inf]")

        Zs = np.array([p.gas.mixture_fraction(fuel,oxidizer) for p in particles])
        Ms = np.array([p.particle_mass for p in particles])
        Ms = Ms / np.sum(Ms)

        # quicksort for cdf
        sorted_id = np.argsort(Zs)
        CDF = np.zeros_like(Ms)
        CDF[sorted_id] = np.cumsum(Ms[sorted_id])

        varZ = np.var(Zs)
        dvar = 0
        Nc = int(Np*max(0.1/sigma_k, 1)) # sigma_k > 0.1, Nc = Np
                                         # sigma_k < 0.1, Nc = Np*0.1/sigma_k
        for i in range(Nc):
            p = int(np.floor(np.random.rand()*Np))
            q = int(np.floor(np.random.rand()*Np))
            d = CDF[p] - CDF[q]
            f = np.exp(-d**2/sigma_k**2/4)
            dvar += 0.5 * f * (Zs[p]-Zs[q])**2 / Nc
        coeff = varZ / dvar
        
        nmix = int(1.5 * omdt * Np * coeff + 1)
        pmix = 1.5 * omdt * Np * coeff / nmix

        for i in range(nmix):
            p = int(np.floor(np.random.rand()*Np))
            q = int(np.floor(np.random.rand()*Np))
            d = CDF[p] - CDF[q]
            f = np.exp(-d**2/sigma_k**2/4)
            if np.random.rand() < f and np.random.rand() < pmix:
                a = np.random.rand()
                phi_pq = (phis[p] + phis[q])/2
                phis[p] += -a*(phis[p] - phi_pq)
                phis[q] += -a*(phis[q] - phi_pq)

    else:
        print("Model %s is not a valid mixing model."%model)
        sys.exit(1)

    # set compositions back to particles
    for i,comp in enumerate(phis):
        particles[i](comp)


def reaction_worker(part_tup):
    """Worker for performing reaction substep given initial state.
    Parameters
    ----------
    part_tup : tuple
        Tuple with mechanism file, temperature, pressure, mass fractions, and time step.
    Returns
    -------
    p : `numpy.array`
        Thermochemical composition of particle following reaction.
    """
    mech, m, T, P, Y, dt = part_tup
    gas = ct.Solution(mech)
    gas.TPY = T,P,Y

    reac = ct.IdealGasConstPressureReactor(gas, volume=m/gas.density)
    netw = ct.ReactorNet([reac])
    netw.advance(netw.time + dt)
    
    return np.hstack((gas.enthalpy_mass, gas.Y))


def reaction_substep(particles, dt, mech):
    """Advance each of the particles in time through reactions.
    Parameters
    ----------
    particles : list of `Particle`
        List of Particle objects to be reacted.
    dt : float
        Time step [s] to increment particles.
    mech : str
        Mechanism filename.
    Returns
    -------
    None
    """
    if not parallel:
        for p in particles:
            p.react(dt)
    else:
        pool = multiprocessing.Pool(processes=Nproc)
        
        # set up a new particle runner for each
        jobs = []
        for p in particles:
            jobs.append([mech, p.particle_mass, p.gas.T, p.gas.P, p.gas.Y, dt])
        jobs = tuple(jobs)

        results = pool.map(reaction_worker, jobs)

        pool.close()
        pool.join()

        # update states of all particles on the main thread
        for i, p in enumerate(particles):
            p(comp=results[i])


def inflow(streams):
    """Determine index of stream for next inflowing particle.
    Parameters
    ----------
    streams : list of `Stream`
        List of Stream objects for inlet streams.
    Returns
    -------
    i_inflow : int
        Index of stream for next inflowing particle.
    """
    # Find stream with largest running flow rate
    sum_flows = 0.0
    fl_max = 0.0
    i_inflow = None
    for i, stream in enumerate(streams):
        streams[i].xflow += stream.flow
        sum_flows += stream.flow

        if streams[i].xflow > fl_max:
            fl_max = streams[i].xflow
            i_inflow = i

    # Check sum of flows
    if sum_flows < 0.0:
        print('Error: sum_flows = {:.4}'.format(sum_flows))
        sys.exit(1)

    # Now reduce running flow rate of selected stream
    streams[i_inflow].xflow -= sum_flows
    return i_inflow


def save_data(idx, time, particles, data):
    """Save temperature and species mass fraction from all particles to array.
    Parameters
    ----------
    idx : int
        Index of timestep.
    time : float
        Current time [s].
    particles : list of `Particle`
        List of `Particle` objects.
    data : `numpy.ndarray`
        ndarray of particle data for all timesteps.
    Returns
    -------
    None
    """
    for i, p in enumerate(particles):
        data[idx, i, 0] = time
        data[idx, i, 1] = p.gas.T
        data[idx, i, 2] = p.gas.P
        data[idx, i, 3:] = p.gas.Y


def parse_input_file(input_file):
    """Parse input file for PaSR operating parameters.
    Parameters
    ----------
    input_file : str
        Filename with YAML-format input file.
    Returns
    -------
    pars : dict
        Dictionary with input parameters extracted from YAML file.
    """

    with open(input_file, 'r') as f:
        pars = yaml.load(f, Loader=Loader)

    case = pars.get('case', None)
    if not case in ['premixed', 'non-premixed']:
        print('Error: mech need to be specified.')
        sys.exit(1)
    if not pars.get('mech', None):
        print('Error: case needs to be one of '
              '"premixed" or "non-premixed".')
        sys.exit(1)
    if not pars.get('temperature', None):
        print('Error: (initial) temperature needs to be specified.')
        sys.exit(1)

    if not pars.get('pressure', None):
        print('Error: pressure needs to be specified.')
        sys.exit(1)

    eq_ratio = pars.get('equivalence ratio', None)
    if not eq_ratio or eq_ratio < 0.0:
        print('Error: eq_ratio needs to be specified and > 0.0.')
        sys.exit(1)

    if not pars.get('fuel', None):
        print('Error: fuel species and mole fraction need to specified.')
        sys.exit(1)

    if not pars.get('oxidizer', None):
        print('Error: oxidizer species and mole fractions '
              'need to be specified.')
        sys.exit(1)
    if not pars.get('sigma_k', None):
        if pars['mixing model'] == "KerM":
            print('Error: for KerM mixing model, sigma_k '
              'need to be specified.')
            sys.exit(1)
        pars['sigma_k'] = None

    # Optional inputs
    if not pars.get('number of particles', None):
        pars['number of particles'] = 100
    if not pars.get('residence time', None):
        pars['residence time'] = 10.e-3
    if not pars.get('mixing time', None):
        pars['mixing time'] = 1.e-3
    if not pars.get('mixing model', None):
        pars['mixing model'] = "IEM"
    if not pars.get('number of residence times', None):
        pars['number of residence times'] = 5

    return pars
