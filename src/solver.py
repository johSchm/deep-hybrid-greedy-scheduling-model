from io import StringIO
import numpy as np
import docplex.cp.utils_visu as visu
from docplex.cp.model import *
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import chain
import pandas as pd
import tensorflow as tf


def sparse_to_dense(x, y, padding_value=-1):
    y = y.numpy()
    x = x.numpy()
    M = tf.shape(y)[1]
    T = tf.shape(y)[-2]
    B = tf.shape(y)[0]
    y_dense = np.empty([B, M, T, 7])
    for i in range(len(y)):
        for m in range(M):
            y_dense[i, m, :] = np.matmul(y[i, m].T, x[i, :, :])
            for t in range(T):
                if y_dense[i, m, t, 0] == 0:
                    y_dense[i, m, t, :2] = (padding_value, padding_value)
    return y_dense.astype(np.float32)


def print_full_df(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    
    
# one hot encoding of labels y
def one_hot_encoder(y, machine_range, job_range):
    M = max(machine_range)
    J = max(job_range)
    y_data_one_hot = np.zeros([len(y), M, J, J])
    for i in tqdm(range(len(y))):
        for m in range(M):
            for j in range(J):
                one_hot = np.zeros(J)
                idx = int(y[i, m, j, 0])
                one_hot[idx] = 1 if idx != -1 else 0
                y_data_one_hot[i, m, j] = one_hot
    y = np.transpose(y_data_one_hot, [0, 1, 3, 2])
    return y


# padding input to get equal input job number
def pad_jobs(x, job_range, nb_setups):
    job_pad = np.append(np.array([-1, -1]), np.zeros(nb_setups))
    padded_x = np.empty([len(list(x)), max(job_range), len(job_pad)])
    padded_x[:] = job_pad
    for i in range(len(x)):
        padded_x[i, :len(x[i]), :] = x[i]
    return padded_x


# padding output to get equal machine number
# and jobs assigned to it
def pad_machines(y, machine_range, job_range):
    job_pad = np.array([-1, -1, -1], np.float32)
    y_padded = np.zeros([len(y), max(machine_range), max(job_range), len(job_pad)])
    y_padded[:] = job_pad
    for i in range(len(y)):
        for m in range(len(y[i])):
            if len(y[i][m]) > 0:
                y_padded[i, m, :len(y[i][m]), :] = y[i][m]
    return y_padded


def y_valid_ms(y):
    return sum(y[:, 0, 0] != -1)
    
    
def check_point_validity(x, y):
    pt = x[:, 0]    
    dd = x[:, 1]
    st = x[:, 2:]
    rd = np.zeros(len(pt), dtype=np.int32)
    nb_m = y_valid_ms(y)
    
    # remove padding 
    pt = pt[pt != -1].astype(np.int32)
    dd = dd[dd != -1].astype(np.int32)
    st = st[np.sum(st, axis=1) != 0, :]
    
    y_hat = solve(pt,dd,rd,nb_m,st)

    y_makespan = np.amax(y[:, :, 1])
    
    rd_y_hat = [np.array(alloc)[:, 2] for alloc in list(y_hat[0].values())]
    y_hat_makespan = max(list(chain(*chain(rd_y_hat))))
    
    return {'y': y_makespan, 'y_solver': float(y_hat_makespan), 'valid': y_makespan <= y_hat_makespan}


def random_validity_check(ds, nb_samples=10):
    ds_x, ds_y = ds
    rand_sample_idxs = np.random.randint(len(ds_x), size=nb_samples)
    results = [check_point_validity(ds_x[idx], ds_y[idx]) for idx in tqdm(rand_sample_idxs)]
    df = pd.DataFrame(results)
    return df


def gen_due_date(max_proc_t,nb_m,nb_t,pt_int=True):
    proc_l=[]
    fac_vec=np.arange(np.ceil(nb_t/2),dtype=np.int64)+1
    fac_vec=np.repeat(fac_vec, 2)
    np.random.shuffle(fac_vec)

    for i in range(nb_t):
        t=np.random.normal(loc=(fac_vec[i]*max_proc_t[i]+max_proc_t[i]/2),scale=max_proc_t[i]/6)
        if pt_int:
            proc_l.append(int(t))
        else:
            proc_l.append(t)
    return np.array(proc_l)


def gen_setup_types(nb_t, nb_s):
    # nb_s is the intended number of different setup tasks
    setup_types = [np.random.randint(0, nb_s - 1) for _ in range(nb_t)]
    setup_types = np.array(setup_types).reshape(-1)
    setup_types = np.eye(nb_s)[setup_types]
    return setup_types


def get_pt(mean, stddev=0.0, size=1):
    if stddev: 
        pt = np.random.normal(loc=mean,scale=stddev,size=size)
        return pt.astype(np.int64)
    return np.fill(mean, size, np.int64)


def simulation(pt_mean,
               pt_std,
               nb_t_range,
               nb_m_range,
               nb_s):
    
    x, y = [], []
    
    nb_m = np.random.randint(
        nb_m_range[0],nb_m_range[1]+1) if type(nb_m_range) is list else nb_m_range
    nb_t = np.random.randint(
        nb_t_range[0],nb_t_range[1]+1) if type(nb_t_range) is list else nb_t_range

    proc_t = np.random.normal(loc=pt_mean,scale=pt_std,size=(nb_t)).astype(np.int64)
    rel_dates = np.zeros(nb_t,dtype=np.int32)
    due_dates = gen_due_date(proc_t,nb_m,nb_t)
    setup_types = gen_setup_types(nb_t, nb_s)

    x.append([{
        'pt': proc_t,
        'dd': due_dates,
        'st': setup_types
    }])

    sol=solve(proc_t, due_dates, rel_dates, nb_m, setup_types)
    sol, model, transitionTimes = sol[:-2], sol[-2], sol[-1]
    
    return x,sol


def solve(proc_t,due_dates,rel_dates,nb_m,setup_types):
    """
    REF: https://ibmdecisionoptimization.github.io/tutorials/html/Scheduling_Tutorial.html
    http://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.model.py.html

    Inspired by House Building Problem, hence the following terms are defined:

    Worker => Machines
    Tasks => Jobs
    Houses => 1 (does not fulfill a purpose here)
    Skills => each machine has the skill to process each job
    Deadline => a day
    """
    NbHouses = 1
    Deadline =  24*60
    Workers = ["M"+str(i) for i in range( nb_m)]
    Tasks = ["T"+str(i) for i in range( proc_t.shape[0])]
    Durations = proc_t
    ReleaseDate = rel_dates
    DueDate = due_dates
    
    Skills=[]
    for w in Workers:
        for t in Tasks:
            Skills.append((w,t,1))

    nbWorkers = len(Workers)
    Houses = range(NbHouses)
    mdl5 = CpoModel()
    tasks = {}
    wtasks = {}
    wseq = {}
    transitionTimes = transition_matrix(len(Tasks))
    for h in Houses:
        for i,t in enumerate(Tasks):
            # add interval decision var for each job, range from 0 to Deadline, and fixed length of PT
            # thus each task has to fit with its pt in the fictional deadline (max time)
            tasks[(h,t)] = mdl5.interval_var(start=[0,Deadline], size=Durations[i])

            # Add transition times between tasks, which do NOT share the same setup time
            for j,t2 in enumerate(Tasks):
                if np.dot(setup_types[i], setup_types[j]) == 0:
                    transitionTimes.set_value(i, j, 10)
                else:
                    transitionTimes.set_value(i, j, 0)

        for i,s in enumerate(Skills):
            # looping over each possible combination of machine and job (skill)
            # add interval decision var for each combi, range from 0 to DD for each job.
            # Thus each job on each machine must be processed within a range of 0 upto its DD.
            wtasks[(h,s)] = mdl5.interval_var(start=[0,DueDate[i%len(Tasks)]],optional=True)
        for w in Workers:
            wseq[w] = mdl5.sequence_var([wtasks[(h,s)] for s in Skills if s[0] == w],
                                        types=[int(s[1][1:]) for s in Skills if s[0] == w ])
    for h in Houses:
        for t in Tasks:
            # add constraint such that if j is in the solution space,
            # then there is exactly one job on a machine.
            mdl5.add( mdl5.alternative(tasks[h,t], [wtasks[h,s] for s in Skills if s[1]==t]) )
            
    for w in Workers:
        # add overlap constraint to enforce transitions is required
        mdl5.add( mdl5.no_overlap(wseq[w], transitionTimes))

    # objective maximize the difference between the due dates and the processing times
    mdl5.add(
        mdl5.maximize(
            mdl5.sum(mdl5.size_of(wtasks[h,s]) - mdl5.end_of(wtasks[h,s])
                     for h in Houses for s in Skills)
        )
    )

    # Solve it
    solver_log_stream = StringIO()
    msol5 = mdl5.solve(log_output=solver_log_stream)

    # transform model solution to a format, which can be handled afterwards
    worker_idx = {w : i for i,w in enumerate(Workers)}
    worker_tasks = [[] for w in range(nbWorkers)]  # Tasks assigned to a given worker
    for h in Houses:
        for s in Skills:
            worker = s[0]
            wt = wtasks[(h,s)]
            worker_tasks[worker_idx[worker]].append(wt)
    sol_dict = {k: [] for k in range(nb_m)}

    for i,w in enumerate(Workers):
        visu.sequence(name=w)
        for k,t in enumerate(worker_tasks[worker_idx[w]]):
            wt = msol5.get_var_solution(t)
            if wt.is_present():
                sol_dict[i].append((k,wt.start,wt.end))
    for i,w in enumerate(Workers):
        sol_dict[i].sort(key=lambda tup: tup[1])
        
    return [sol_dict, msol5, transitionTimes]