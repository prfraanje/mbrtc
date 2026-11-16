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
#   ./demo_control_simulator_deadbeat.py  # linux/mac system terminal
#   demo_control_simulator_deadbeat.py    # windows cmd or similar terminal
#   python3 demo_control_simulator_deadbeat.py
#
# When running from spyder take following points into account:
#  - Configure IPython to not inline graphics, under:
#       Tools -> Preferences -> IPython console -> Graphics -> Graphics backend
#  - Configure to run in external system terminal to prevent conflicts with
#    the asyncio event loop that spyder is already using:
#       Tools -> Preferences -> Run -> tick Execute in an external system terminal
#
# Rufus Fraanje, p.r.fraanje@hhs.nl
# Date: 13/11/2020
# License: GPLv3

#
# nest_asyncio may help to run in spyder, but for me only running
# in external system terminal works
# import nest_asyncio
# nest_asyncio.apply()

import asyncio

from time import time_ns

from math import pi, cos, sin

import matplotlib.pyplot as plt

import numpy as np

from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout

from pygments.lexers.python import Python3Lexer

# process simulation task (more precisely: coroutine, same for 'tasks' below)
async def process(shared_dict,dt=0.005):
    """Simulation of the process, each iterations takes dt seconds.
       shared_dict is a dictionary that is shared by the various tasks.
       The process runs while shared_dict['run'] is True, and then it
       reads the value of the current control from shared_dict['u']
       and calculates the corresponding output and writes that in
       shared_dict['y']."""
    #theta = 1 #pi-0.1
    #omega = 1
    time_now = time_ns()
    u = 0
    y_p1 = 1
    y = 1
    while shared_dict['run']:
        u_p1,u = u,shared_dict['u']
        y_p1,y = y,shared_dict['y']
        # pendulum
        #dx1_dt = x2
        #dx2_dt = -sin(x1) + u * cos(x1)
        # double integrator: pure mass (linear / rotational)
        #domega_dt = u
        # note: the following lines are crude extrapolation!
        #theta += dt * omega
        #omega += dt * domega_dt
        y_next = 2 * y - y_p1 + (dt**2)/2 * (u + u_p1)
        shared_dict['y'] = y_next
        time_prev, time_now = time_now, time_ns()
        shared_dict['sample_time'] = (time_now - time_prev)/1e6
        await asyncio.sleep(dt) # wait some time, while other tasks can be done

        
# controller task
async def control(shared_dict,dt=0.005):
    """Simulation of the control task, each iterations takes dt seconds.
       shared_dict is a dictionary that is shared by the various tasks.
       The process runs while shared_dict['run'] is True, and when
       shared_dict['control'] is false it writes 0 as control value
       else then it calculates the control value from the value shared_dict['y'].
       The control value u is finally written in shared_dict['u']."""
    # note, if dt is changed, following should change as well!
    h2 = dt*dt
    r1 = 0.75
    s0 = 2.5/h2
    s1 = -1.5/h2
    t0 = 1/(2*h2)
    t1 = 0.
    u = 0.
    uc = 0.
    y = 0.
    shared_dict['dt'] = dt
    while shared_dict['run']:
        if shared_dict['control']:
            y_prev, y = y, shared_dict['y']
            uc_prev, uc = uc, shared_dict['r']  # command / reference
            u_prev = u
            u = t0 * uc + t1 * uc_prev - s0* y - s1 * y_prev - r1 * u_prev
        else:
            u = 0.
        shared_dict['u'] = u
        dt = shared_dict['dt']
        await asyncio.sleep(dt)
        
    
            

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
                                  'u', 'y', 'plot_buffer'])
    print('Enter your single line command and press return, enter \"stop\" to exit.\n For example: shared_dict[\'r\'] = 1 or enter just shared_dict to see it\'s current value')
    while shared_dict['run']:
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
    while shared_dict['run']:
        if shared_dict['print']:
            print(f"run = {shared_dict['run']}, control = {shared_dict['control']}, y = {shared_dict['y']}, u = {shared_dict['u']}")
        await asyncio.sleep(dt)

        
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
        self._buffer = np.zeros([2*self._length,*self._dims_sample])
        self._last = self._length
        
    def get(self):
        """Return the length samples in the numpy array buffer. """
        return self._buffer[self._last-self._length:self._last]
    
    def update(self,sample):
        """Update the cyclic buffer with the sample. Sample is a list or a numpy array
        which dimensions should match with dims_sample set by the __init__ method."""
        last = self._last     # pointer to new place
        length = self._length
        # store sample at position last and last-length:
        self._buffer[last] = self._buffer[last-length] = np.array(sample)
        last += 1
        if last>=2*length:
            last = length
        self._last = last

        
# update plot buffer task
async def update_plot_buffer(shared_dict,dt=0.005):
    """Task to update the cyclic buffer periodically with an interval of dt seconds."""
    while shared_dict['run']:
        sample = [shared_dict['y'], shared_dict['u'],shared_dict['sample_time']]
        shared_dict['plot_buffer'].update(sample)
        await asyncio.sleep(dt)

        
# plotter task
async def plotter(shared_dict,dt=0.05):
    """Task to plot the samples in the cyclic buffer periodically with an interval of dt seconds."""
    plt.ioff()  # do not make the plot interactive, else plot not updated correctly
                # or it takes the focus from the prompt to the figure which is quite annoying.
    fig,axes = plt.subplots(3,1)
    plt.show(block=False)
    while shared_dict['run']:
        if shared_dict['plot']:
            [axes[i].cla() for i in range(3)]  # clear all axes
            data = shared_dict['plot_buffer'].get()
            axes[0].plot(data[:,0])
            axes[0].set_ylabel('y')
            axes[1].plot(data[:,1])
            axes[1].set_ylabel('u')
            y = shared_dict['y']
            axes[2].plot([0,sin(y)],[0,-cos(y)],lw=2)
            axes[2].set_xlim([-4,4])
            axes[2].set_ylim([-1.2,1.2])
            axes[2].grid(True)
            #axes[3].plot(data[:,2])
            fig.canvas.draw()         # this and following line to 
            fig.canvas.flush_events() # update figure
            
        await asyncio.sleep(dt)
    plt.close(fig) # at end of task, close figure

    
# main function:
async def main():
    """Main task, for initialization and gathering all the tasks."""
    shared_dict = {}
    shared_dict['run'] = True         # when set to False, will stop simulation
    shared_dict['control'] = True     # to control or not
    shared_dict['print'] = False      # to print or not
    shared_dict['plot'] = True        # to plot or not
    shared_dict['r'] = 0.             # reference value to control to
    shared_dict['u'] = 0.             # initial value control signal
    shared_dict['y'] = 0.
    shared_dict['sample_time'] = 0.
    buffer_size = 100                 # number of samples in plot
    dims_sample = (3,)
    # next line, create fast FIFO cyclic buffer to store last buffer_size samples
    shared_dict['plot_buffer'] = CyclicBuffer(buffer_size,dims_sample)

    # tasks to perform concurrently (just comment tasks you don't need):
    await asyncio.gather(
         process(shared_dict),
         control(shared_dict),
         repl(shared_dict),
         printer(shared_dict),
         update_plot_buffer(shared_dict),
         plotter(shared_dict),
    )

    
# Only run the tasks when run as a script (rather than a module)
if __name__ == '__main__':
    asyncio.run(main())
