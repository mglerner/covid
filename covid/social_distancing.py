import numpy as np, scipy as sp, pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import animation
from IPython.display import HTML
from itertools import combinations
from collections import namedtuple

"""
Units: time is measured in 12 hour chunks, but there's no strict correspondence to units here.
"""

class EffectiveArea:
    """Tells us the boundaries of the area that people can live in.
    """
    def __init__(self,):
        self.xmin, self.xmax = 0, 20
        self.ymin, self.ymax = 0, 20

class Person:
    """
    In this version, after 14 days, you either recover or die
    """
    def __init__(self,effectivearea,state='succeptible',distancing=False):
        self.r = 0.15
        self.days_infected = 0
        self._typical_recovery_days = 14
        self.mortality_rate = {True:0.05,False:0.02} # index is whether we're over capacity
        self.distancing = distancing
        if self.distancing:
            self.m = 1000
        else:
            self.m = 1
        self.state = state
        if self.state == 'infected':
            self.days_infected = np.random.randint(self._typical_recovery_days)
        self.ea = effectivearea
        self.x, self.y = np.random.uniform(self.ea.xmin,self.ea.xmax), np.random.uniform(self.ea.ymin,self.ea.ymax)
        if self.distancing:
            self.distance(force=True)
        else:
            self.undistance(force=True)
    def distance(self,force=False):
        if force or not self.distancing:
            self.m = 1000
            self.vx, self.vy = 0,0
            self.distance = True
    def undistance(self,force=False):
        if force or self.distancing:
            self.m = 1
            self.vx, self.vy = np.random.normal(size=2) # Maxwell Boltzmann?
            self.distancing = False
    def move(self,dt,ea,overcapacity=False):
        self.x, self.y = self.x + self.vx*dt, self.y + self.vy*dt
        
        """People don't recover instantly. Once it's been two weeks, 
        they start getting a chance to recover. We could also give 
        them a chance to die here.
        """
        if self.state == 'infected':
            if self.days_infected < self._typical_recovery_days:
                self.days_infected += dt
            else:
                if np.random.uniform() < self.mortality_rate[overcapacity]:
                    self.state = 'dead'
                else:
                    self.days_infected = 0
                    self.state = 'recovered'
    
def collide(p1,p2):
    """This is the 2D elastic collision problem from your intro physics book.
    """
    if p1.state == 'infected' and p2.state == 'succeptible':
        p2.state = 'infected'
    elif p2.state == 'infected' and p1.state == 'succeptible':
        p1.state = 'infected'

    m1, m2 = p1.m, p2.m
    r1, r2 = np.array([p1.x,p1.y]), np.array([p2.x,p2.y])
    v1, v2 = np.array([p1.vx,p1.vy]), np.array([p2.vx,p2.vy])
    M = m1 + m2
    d = np.linalg.norm(r1 - r2)**2
    u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
    u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
    p1.vx,p1.vy = u1
    p2.vx,p2.vy = u2

    
class Universe:
    def __init__(self,
                 npeople, # How many people there are in the world
                 initial_infection_chance=0.1, # Initial fraction of population which is infected
                 distancing=0.0, # Fraction of population which practices distancing
                 hospital_beds_percent = 1.0 # Better is 0.003 which is closer to reality
                ):
        self.npeople = npeople
        self.ea = EffectiveArea()
        self.dt = 0.1
        self.hospital_beds_percent = hospital_beds_percent
        self.hospital_beds = int(npeople*hospital_beds_percent)
        self.data = None # gets set in self.run
        def _state():
            if np.random.uniform() < initial_infection_chance:
                return 'infected'
            return 'succeptible'
        def _distancing():
            return np.random.uniform() < distancing
        self.people = [Person(self.ea,_state(),_distancing()) for i in range(self.npeople)]
        # self.color = {'succeptible':0.5,'infected':0.0,'recovered':0.7} # old color scheme
        self.color = {'succeptible':'lightblue','infected':'red','recovered':'green','dead':'brown'}
        
    def _step(self):
        """iterate through one timestep
        """
        points = list(zip([p.x for p in self.people],[p.y for p in self.people]))
        dists = euclidean_distances(points,points)
        close = dists < 2*self.people[0].r
        close = close.tolist()
        for (i,j) in combinations(range(self.npeople),2):
            if close[i][j]: # a bit faster than numpy indexing once things get big.
                collide(self.people[i],self.people[j])
                
        # Are we over capacity?
        ninfected = len([p for p in self.people if p.state == 'infected'])
        # about 5% need hospitalization
        overcapacity = 0.05 * ninfected > self.hospital_beds
        for p in self.people:
            p.move(self.dt,self.ea,overcapacity)
            if p.x <= self.ea.xmin or p.x >= self.ea.xmax: 
                p.vx = -p.vx
            if p.y <= self.ea.ymin or p.y >= self.ea.ymax:
                p.vy = -p.vy
    
    def run(self,steps,stop_distancing_at = None):
        """Run a simulation
        
        Internal data looks like
        
        ```
        x_coords[frame,particle_number]
        ```
        """
        x_coords = np.zeros((steps,len(self.people)))
        y_coords = np.zeros((steps,len(self.people)))
        state = np.zeros((steps,len(self.people)),dtype='object')
        
        # SIR model. `i` is a questionable variable name TBH
        s,i,r,d = np.zeros(steps),np.zeros(steps),np.zeros(steps),np.zeros(steps)
        def pop_count(people):
            s,i,r,d = 0,0,0,0
            for p in people:
                if p.state == 'succeptible':
                    s += 1
                elif p.state == 'infected':
                    i += 1
                elif p.state == 'recovered':
                    r += 1
                elif p.state == 'dead':
                    d += 1
                    
            return s,i,r,d
        s[0],i[0],r[0],d[0] = pop_count(self.people)
        x_coords[0] = [p.x for p in self.people]
        y_coords[0] = [p.y for p in self.people]
        state[0] = [p.state for p in self.people]
        
        for step in range(1,steps):
            if step == stop_distancing_at:
                for p in self.people:
                    p.undistance()
            self._step()
            s[step],i[step],r[step],d[step] = pop_count(self.people)
            x_coords[step] = [p.x for p in self.people]
            y_coords[step] = [p.y for p in self.people]
            state[step] = [p.state for p in self.people]
        dtype = namedtuple('RunData',['x','y','state','s','i','r','d','steps'])
        self.data = dtype(x_coords,y_coords,state,s,i,r,d,steps)
        
    def draw(self,ax=None):
        """
        A very simple method to draw the current state. Better graphing comes in the
        animation functions.
        """
        if ax is None:
            fig,ax = plt.subplots(figsize=(5,5))
        scat = ax.scatter([p.x for p in self.people],[p.y for p in self.people],
                   c = [self.color[p.state] for p in self.people],
                   marker='.')
        return scat,



def getanim(u):
    fig,ax = plt.subplots(figsize=(4,4))

    left, width = 0.1, 0.85
    bottom, height = 0.1, 0.65
    spacing = 0.02

    rect_universe = [left, bottom, width, height]
    rect_trend = [left, bottom + height + spacing, width, 0.2]

    ax_universe = plt.axes(rect_universe)
    ax_universe.tick_params(axis='x', which='both',bottom=False,top=False,labelbottom=False)
    ax_universe.axis('off')
    ax_trend = plt.axes(rect_trend)

    s,i,r,d = u.data.s,u.data.i,u.data.r,u.data.r

    ax_trend.stackplot(range(len(s)), i, s, r, d, labels=['sick','healthy','recovered','dead'],
                       colors=[u.color['infected'],u.color['succeptible'],u.color['recovered'],u.color['dead']])

    scat, = u.draw(ax_universe)
    
    def drawframe(i):
        data = np.column_stack(([u.data.x[i],u.data.y[i]]))
        scat.set_offsets(data)
        colors = np.array([u.color[_] for _ in u.data.state[i]])
        scat.set_color(colors)

        _s,_i,_r = np.zeros(len(u.data.s)),np.zeros(len(u.data.s)),np.zeros(len(u.data.s))
        _s[:i],_i[:i],_r[:i] = u.data.s[:i],u.data.i[:i],u.data.r[:i]
        #ax_trend.stackplot(range(len(_s)), _s, _i, _r, labels=['s','i','r'],colors=['blue','green','yellow'])
        #ax_trend.legend(loc='upper left')

        return scat,


    anim = animation.FuncAnimation(fig, drawframe, frames=u.data.steps,
                                  interval=20, blit=False, repeat=False)
    return anim



def draw_stacked_plot(u):
    s,i,r,d = u.data.s,u.data.i,u.data.r,u.data.d
    plt.stackplot(range(len(s)), i, s, r, d, labels=['sick','healthy','recovered','dead'],
                           colors=[u.color['infected'],u.color['succeptible'],u.color['recovered'],u.color['dead']])
    plt.legend(loc='center left')
    plt.hlines(y=u.hospital_beds*20,xmin=0,xmax=len(s),linestyle='dashed')
    return None
