"""
MOR-Phyiscs

morp.py

collection of parameterized operators
"""

import tensorflow as tf
import numpy as np
import itertools
import toolz
import random

 
#sine and cosine transforms
@tf.function
def rfft(u):
    uh = tf.signal.fft(tf.complex(u,tf.constant(0.,tf.float64
                                     )))[...,0:int(u.shape[-1]//2)+1]
    return uh

@tf.function
def irfft(uh):
    u = tf.math.real(tf.signal.ifft(tf.concat([uh,
               tf.reverse(tf.math.conj(uh[...,1:-1]),axis=[-1])],-1)))
    return u

@tf.function
def st(v):
    s = v.shape[-1]+1
    vh = tf.math.imag(rfft((tf.concat([-tf.reverse(v,[-1]),v],-1)))
          * np.exp(1.j*(np.arange(s) * 2*np.pi * -.5/v.shape[-1]/2 )))
    return vh
@tf.function
def ist(vh):
    s = vh.shape[-1]
    v = -tf.reverse(irfft(tf.complex(tf.constant(0.,
                  dtype=tf.float64),vh) 
          * np.exp(1.j*(np.arange(s) 
          * 2*np.pi * .5/(2*(vh.shape[-1]-1)) )))[...,0:s-1],[-1])
    return v
@tf.function
def ct(phi):
    s = phi.shape[-1]+1
    phih = tf.math.real(rfft((tf.concat([tf.reverse(phi,[-1]),
                   phi],-1))) * np.exp(1.j*(np.arange(s) 
          * 2*np.pi * -.5/phi.shape[-1]/2 )))
    return phih
@tf.function
def ict(phih):
    s = phih.shape[-1]
    phi = tf.reverse(irfft(tf.complex(phih,tf.constant(0.,
                   dtype=tf.float64)) * np.exp(1.j*(np.arange(s)
          * 2*np.pi * .5/(2*(phih.shape[-1]-1)) )))[...,0:s-1],[-1])
    return phi




#neural networks
class nn:
    def __init__(self,width,depth,dimi,dimo):
        layers = (
                     [tf.keras.layers.Input(dimi,dtype=tf.float64)]
                    +[tf.keras.layers.Dense(width,activation='elu',
                            dtype=tf.float64) for _ in range(depth-1)]
                    +[tf.keras.layers.Dense(dimo,dtype=tf.float64)]
                 )
        self.func  = tf.keras.Sequential(layers)
        self.vlist = self.func.variables
    @tf.function
    def __call__(self,u):
        u2 = self.func(u)
        return u2

#neural networks - takes as input a vector of parameters in addition to the state and wave-vector
class nnp:
    def __init__(self,width,depth,dimi,dimo,dimp):
        input1 = tf.keras.layers.Input(shape=(None,dimi,))
        input2 = tf.keras.layers.Input(shape=(None,dimp,))
        x1 = tf.keras.layers.Dense(width)(input1)
        x2 = tf.keras.layers.Dense(width)(input2)
        added = tf.nn.elu(tf.keras.layers.Add()([x1, x2]))
        layers = (
                     [tf.keras.layers.Dense(width,activation='elu',
                            dtype=tf.float64) for _ in range(depth-2)]
                    +[tf.keras.layers.Dense(dimo,dtype=tf.float64)]
                 )
            
        self.func  = tf.keras.models.Model(inputs=[input1, input2],
                         outputs=tf.keras.Sequential(layers)(added))
        self.vlist = self.func.variables
    @tf.function
    def __call__(self,u,params):
        u2 = self.func([u,params])
        return u2



#operator - cosine basis
class Leven:
    def __init__(self,depth,width,dimi,dimp,k,cons=False):
        self.nn = nnp(depth,width,dimi,1,dimp)
        self.nn2 = nnp(depth,width,1,1,dimp)
        self.vlist = self.nn.vlist + self.nn2.vlist
        self.k = k
        self.p = np.ones(len(k))
        if cons:
            self.p[0] = 0.
    @tf.function
    def __call__(self,u,params):
        hu = self.nn(u,params)[...,0]
        huh = ct(hu)
        huh2 = huh * self.nn2(self.k,params)[...,0] * self.p
        phi2 = ict(huh2)
        return phi2

#operator - sine basis
class Lodd:
    def __init__(self,depth,width,dimi,dimp,k):
        self.nn = nnp(depth,width,dimi,1,dimp)
        self.nn2 = nnp(depth,width,1,1,dimp)
        self.vlist = self.nn.vlist + self.nn2.vlist
        self.k = k
    @tf.function
    def __call__(self,u,params):
        hu = self.nn(u,params)[...,0]
        huh = st(hu)
        huh2 = huh * self.nn2(self.k,params)[...,0]
        v2 = ist(huh2)
        return v2


#operator - 2d fourier basis
class L2d:
    def __init__(self,depth,width,k,cons=False,sym=False,iso=False):
        self.nn = nn(depth,width,1,1)
        if sym:
            self.h = lambda u: tf.sign(u)  \
                        *self.nn(tf.expand_dims(
                          tf.expand_dims(tf.abs(u),-1),-1))[...,0,0]
        else:
            self.h = lambda u: self.nn(tf.expand_dims(
                          tf.expand_dims(u,-1),-1))[...,0,0]
        if iso:
            self.nn2 = nn(depth,width,1,2)
            self.k = tf.expand_dims(
                     tf.expand_dims(np.linalg.norm(k,axis=-1),-1),-1)
        else:
            self.nn2 = nn(depth,width,2,2)
            self.k = tf.expand_dims(k,-2)
        self.p = np.ones(k.shape[0:-1])
        if cons:
            self.p[0,0] = 0.
        self.vlist = self.nn.vlist + self.nn2.vlist
    @tf.function
    def __call__(self,u):
        gk = self.nn2(self.k)[...,0,:]
        hu = self.h(u)
        huh = tf.signal.rfft2d(hu)
        Luh = self.p * tf.complex(gk[...,0],gk[...,1]) * huh
        Lu = tf.signal.irfft2d(Luh)
        return Lu


#operator - 1d fourier basis
class L1d:
    def __init__(self,depth,width,k,cons=False,sym=False):
        self.nn = nn(depth,width,1,1)
        if sym:
            self.h = lambda u: tf.sign(u)*self.nn(
                    tf.expand_dims(tf.expand_dims(
                            tf.abs(u),-1),-1))[...,0,0]
        else:
            self.h = lambda u: self.nn(tf.expand_dims(
                        tf.expand_dims(u,-1),-1))[...,0,0]
        self.nn2 = nn(depth,width,1,2)
        self.k = tf.expand_dims(k,-2)
        self.p = np.ones(k.shape[0:-1])
        if cons:
            self.p[0] = 0.
        self.vlist = self.nn.vlist + self.nn2.vlist
    @tf.function
    def __call__(self,u):
        gk = self.nn2(self.k)[...,0,:]
        hu = self.h(u)
        huh = tf.signal.rfft(hu)
        Luh = self.p * tf.complex(gk[...,0],gk[...,1]) * huh
        Lu = tf.signal.irfft(Luh)
        return Lu
