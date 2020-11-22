#!/usr/bin/env python3

# Simulation of discrete-time real-time dynamic control systems
# with zero-order hold (zoh) discrete time control.
# It is based on concurrent tasks defined by async coroutines
# Python 3.7 or higher is necessary (an alternative for Python 3.5
# is available upon request, however working with most recent stable
# versions is recommended).
#
# Dependencies:
#    - asyncio (standard library)
#    - math (standard library)
#    - matplotlib (only needed for plotting)
#    - numpy (only for data buffering and plotting)
#    - prompt_toolkit (for keyboard interaction, essential)
#    - pygments (for syntax coloring with prompt_toolkit)
#
# run with:
#   ./demo_control_simulator.py  # linux/mac system terminal
#   demo_control_simulator.py    # windows cmd or similar terminal
#   python3 demo_control_simulator.py
# or run from a python shell with:
# >>> exec(open('demo_control_simulator.py').read())
#
# running from ipython3 or jupyter console may give strange results, most likely
# because of conflicts with asyncio and/or getting the event loop.
#
# When running from spyder take following points into account:
#  - Configure IPython to not inline graphics, under:
#       Tools -> Preferences -> IPython console -> Graphics -> Graphics backend
#  - Configure to run in external system terminal to prevent conflicts with
#    the asyncio event loop that spyder is already using:
#       Tools -> Preferences -> Run -> tick Execute in an external system terminal
#
# When running in thonny, just execute by pressing Ctrl+T
#
# Rufus Fraanje, p.r.fraanje@hhs.nl
# Date: 21/11/2020
# License: GPLv3

import asyncio
from asyncio import sleep # quicker access, because its intensively used

# Plotting is deferred to a separate process, data is send over a Queue
# (a Pipe can be used here as well, but Queue has better buffering and allows more control,
# e.g. for a Pipe we cannot limit size, such that it may be overloaded when data is put in
# the pipe at a higher rate than it is consumed by the plotting process.)
from multiprocessing import Process, Queue, queues

import time
from time import time_ns

from math import pi, cos, sin

import matplotlib.pyplot as plt

import numpy as np
from numpy import array # for quick access to np.array in 'real-time' code

from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout

from pygments.lexers.python import Python3Lexer

class Sentinel():
    """Sentinel class for sending stop signal over e.g. a queue.
       Use isinstance(obj,Sentinel) to determine if obj is the sentinel.
       Just create the sentinel by calling: Sentinel().
    """

# State-space model class:
class StateSpaceModelDiscreteTime:
    def __init__(self,A,B,C,D,x0,dt):
        """Class for definition of Discrete Time State-Space Model:
            x_{k+1} = A x_k + B u_k
            y_k     = C x_k + D u_k
        with x_k the state, u_k the input and y_k the output at some
        time instant k, dt is the time between two subsequent time instants.
        """
        self.A = A    # state-transition matrix
        self.B = B    # input matrix
        self.C = C    # output matrix
        self.D = D    # direct feedthrough
        self.x = x0   # initial state
        self.dt = dt  # sampling time, sometimes written as h

    def run(self,u):
        """Perform one iteration of the state-space model, 
        updates the state self.x and returns the output y.
        """
        # note A @ x is matrix-vector multiplication, one can
        # write the same as np.dot(A,x) or A.dot(x)
        y = self.C @ self.x + self.D @ u      # first compute output
        self.x = self.A @ self.x + self.B @ u # then update state
        return y

    def __call__(self,u):
        """Run one iteration of state-space model, see self.run().
           When model is a StateSpaceModel, model(u) performs the 
           update and returns the output, equivalent to model.run(u).
        """
        return self.run(u)

    
# process simulation task (more precisely: coroutine, same for 'tasks' below)
async def process(shared_dict,dt=0.001):
    """Simulation of the process, each iterations takes dt seconds.
       shared_dict is a dictionary that is shared by the various tasks.
       The process runs while shared_dict['run'] is True, and then it
       reads the value of the current control from shared_dict['u']
       and calculates the corresponding output and writes that in
       shared_dict['y']."""
    # example A.1 Double Integrator, from the book:
    # Åström and Wittenmark, Computer-Controlled Systems -- Theory and Design.
    h = dt
    A = np.array([[1., h],[0., 1.]])
    B = np.array([[h**2/2],[h]])
    C = np.array([[1., 0.]])
    D = np.array([[0.]])
    x0 = np.array([[1.],[0.]])
    model = StateSpaceModelDiscreteTime(A,B,C,D,x0,h)
    time_now_ns = time_ns()
    while True:
        if not shared_dict['run']: break
        time_prev_ns, time_now_ns = time_now_ns, time_ns()    # times in ns
        shared_dict['dt'] = (time_now_ns - time_prev_ns)/1e6  # delta time in ms
        u = shared_dict['u']
        y = model(array([[u]]))[0,0]  # bit awkward but we need 1x1 numpy matrix for a scalar
        shared_dict['y'] = y
        dt_remaining = max(0,dt - (time_ns() - time_now_ns)/1e9)
        await sleep(dt_remaining)   # wait some time, while other tasks can be done
        
        
# controller task
async def control(shared_dict,dt=0.05):
    """Simulation of the control task, each iterations takes dt seconds.
       shared_dict is a dictionary that is shared by the various tasks.
       The process runs while shared_dict['run'] is True, and when
       shared_dict['control'] is false it writes 0 as control value
       else then it calculates the control value from the value shared_dict['y'].
       The control value u is finally written in shared_dict['u']."""
    # put dt in a the shared_dict as ctrl_dt, so that it can be changed on the fly: 
    shared_dict['ctrl_dt'] = dt
    u = 0.
    uc = 0.
    y = 0.
    while True:
        if not shared_dict['run']: break
        time_now_ns = time_ns()
        if shared_dict['control']:
            dt = shared_dict['ctrl_dt']  # read dt to allow changes on the fly
            # Deadbeat controller, set/update controller parameters
            h2 = dt*dt
            r1 = 0.75
            s0 = 2.5/h2
            s1 = -1.5/h2
            t0 = 1/(h2)
            t1 = 0.
            y_prev, y = y, shared_dict['y']
            uc_prev, uc = uc, shared_dict['r']  # command / reference
            u_prev = u
            u = t0 * uc + t1 * uc_prev - s0 * y - s1 * y_prev - r1 * u_prev
        else:
            u = 0.
        shared_dict['u'] = u
        dt_remaining = max(0,dt - (time_ns() - time_now_ns)/1e9)
        await sleep(dt_remaining)
            

# command line repl (read, eval, print loop) task for user interaction
async def repl(shared_dict):
    """REPL task that provides a prompt to enter python commands, and provides
    access to shared_dict for inspection but also for adjusting values. It evaluates
    or executes the code and shows the result if any. It both deals with expressions
    (code that gives a result), such as:
    shared_dict['u']   # which shows the value of the current control
    and statements, such as:
    shared_dict['print'] = True  # this will start the printer task (see below).
    Note, that while at the prompt waiting and writing input the other tasks are 
    being executed, thus providing concurrent behavior.
    """
    session = PromptSession(multiline=False)
    # Configure the completer with specifying some words to be completed
    # by the prompt in the repl task (see below).
    my_completer = WordCompleter(['shared_dict', 'stop', 'False', 'True',
                                  'print', 'run', 'control', 'plot',
                                  'u', 'y', 'plot_buffer',
                                  'ctrl_dt'])
    print('Enter your single line command and press return, enter \"stop\" to exit.')
    while True:
        if not shared_dict['run']: break
        with patch_stdout(): # to prevent screen clutter at the prompt line
            res = await session.prompt_async(ANSI('\x1b[01;34m-> '),
                                             lexer=PygmentsLexer(Python3Lexer),
                                             completer=my_completer,
                                             auto_suggest=AutoSuggestFromHistory())
        if res == 'stop': # quick way to stop
            shared_dict['run'] = False
        else:
            try: # first try to evaluate expression
                result = eval(res)
                if result is not None:  # print result when not trivial
                    print(result)
            except SyntaxError:
                try: # when not an expression, try to execute statement
                    exec(res)
                except Exception as e:  # else print exceptions
                    print(e)
            except Exception as e:      # idem
                print(e)

   
# status printer task
async def printer(shared_dict,dt=1):
    """printer task to shows some values in the shared_dict at a rate of dt seconds."""
    while True:
        if not shared_dict['run']: break
        if shared_dict['print']:
            print(f"run = {shared_dict['run']}, control = {shared_dict['control']}, y = {shared_dict['y']}, u = {shared_dict['u']}")
        await sleep(dt)

        
# fast cyclic fifo buffer class for storing signal data
# (writes samples 2 times, so its double length,
# sacrificing some memory to prevent time-consuming memory shifting)
class CyclicBuffer(object):
    """Implementation of a fast cyclic FIFO (first in first out) buffer."""
    def __init__(self,length=1,dims_sample=(1,)):
        """Create the cyclic buffer for length (integer) samples, where dims_sample is a 
           tuple specifying the dimensions of each sample. So when a sample is an array
           of 4 elements dims_sample=(4,), but also higher dimensional data structures such
           as matrices (dims_sample has 2 elements) and tensors (dims_sample more than 
           2 elements) are allowed."""
        self._length = length
        self._dims_sample = dims_sample
        self._buffer = np.zeros((2*self._length,*self._dims_sample))
        self._last = 0
        
    def get(self):
        """Return the length samples in the numpy array buffer. """
        return self._buffer[self._last:self._last+self._length]
    
    def update(self,sample):
        """Update the cyclic buffer with the sample. Sample is a list or a numpy array
        which dimensions should match with dims_sample set by the __init__ method."""
        last = self._last     # pointer to new place
        length = self._length
        # store sample at position last and last-length:
        self._buffer[last] = self._buffer[last+length] = sample
        self._last = (last + 1) % length

    def __call__(self):
        """Rather than buffer.get() you can get the buffer by buffer() as well."""
        return self.get()

        
# update plot buffer task
async def update_plot_buffer(shared_dict,dt=0.005):
    """Task to update the cyclic buffer periodically with an interval of dt seconds."""
    while True:
        if not shared_dict['run']: break
        sample = [shared_dict['y'], shared_dict['u'], shared_dict['dt']]
        shared_dict['plot_buffer'].update(sample)
        await sleep(dt)

        
# plotter task
async def plotter(plot_queue,shared_dict,dt=0.1):
    """Task to plot the samples in the cyclic buffer periodically with an interval of dt seconds.
       To save time, the data is send over the plot_queue to another process that does the actual
       plotting, see the plot_process function.
    """
    while True:
        if not shared_dict['run']: break
        if shared_dict['plot']:
            data = shared_dict['plot_buffer'].get()
            try:
                plot_queue.put_nowait(data)
            except queues.Full:
                pass
        await sleep(dt)
    plot_queue.put(Sentinel()) # send the sentinel to inform plot_process to stop

    
def plot_process(plot_queue):
    """plot_process listens to the plot_queue and plots the data received."""
    plt.ioff()  # do not make the plot interactive, else plot not updated correctly
                # or it takes the focus from the prompt to the figure which is quite annoying.
    fig,axes = plt.subplots(4,1)
    plt.show(block=False)
    while True:
        [axes[i].cla() for i in range(4)]  # clear all axes
        try:
            data = plot_queue.get(timeout=0.1)
            if isinstance(data,Sentinel):
                break  # jump out of while-loop to close figure and stop plot_process
            axes[0].plot(data[:,0])
            axes[0].set_ylabel('y')
            axes[1].plot(data[:,1])
            axes[1].set_ylabel('u')
            y = data[-1,0]  # take most recent value of y
            axes[2].plot([0,sin(y)],[0,-cos(y)],lw=2)
            axes[2].set_xlim([-4,4])
            axes[2].set_ylim([-1.2,1.2])
            axes[2].grid(True)            
            axes[3].plot(data[:,2])
            fig.canvas.draw()         # this and following line to 
            fig.canvas.flush_events() # update figure
        except queues.Empty:
            time.sleep(0.1)
    plt.close(fig)
    
# main function:
async def main(plot_queue):
    """Main task, for initialization and gathering all the tasks."""
    shared_dict = {}
    shared_dict['run'] = True         # when set to False, will stop simulation
    shared_dict['control'] = True     # to control or not
    shared_dict['print'] = False      # to print or not
    shared_dict['plot'] = True        # to plot or not
    shared_dict['r'] = 0.             # reference value to control to
    shared_dict['u'] = 0.             # initial value control signal
    buffer_size = 500                 # number of samples in plot
    dims_sample = (3,)
    # next line, create fast FIFO cyclic buffer to store last buffer_size samples
    shared_dict['plot_buffer'] = CyclicBuffer(buffer_size,dims_sample)
    
    # tasks to perform concurrently (just comment tasks you don't need):        
    await asyncio.gather(
        process(shared_dict),
        control(shared_dict),
        repl(shared_dict),
#        printer(shared_dict),
        update_plot_buffer(shared_dict),
        plotter(plot_queue,shared_dict),
    )
    

# Only run the tasks when run as a script (rather than a module)
if __name__ == '__main__':
    plot_queue = Queue(maxsize=1)  # no need to buffer more than 1 item
    plot_p = Process(target=plot_process,args=(plot_queue,)) # plotting is done in separate process
    plot_p.start()         # start the plot_process
    asyncio.run(main(plot_queue))  # run the main task, will finish if shared_dict['run'] = False
    plot_p.join()          # only proceed when plot_process is finished
    plot_queue.close()     # close the plot_queue
    print('Done')
    

# References:
# - Karl. J. Åström and Björn Wittenmark, Computer-Controlled Systems -- Theory and Design, 3rd edition, Dover Publications, Inc., 2011.
# Excellent reference on Python (e.g. Ch. 20 on multiprocessing Process and Queue is):
# - David M. Beazley, Python -- Essential Reference, 4th edition, Pearson Education, 2009. (we look forward to the 5th edition with complete update for python 3, that appears in 2021)
# Good references for async/await and asyncio are:
# - Luciano Ramalho, Fluent Python -- Clear, Concise, and Effective Programming, O'Reilly, 2015.
# - Gabriele Lanaro, Quan Nguyen and Sakis Kasampalis, Advanced Python Programming -- Learning path, Packt, 2019.
