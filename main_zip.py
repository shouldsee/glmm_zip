#!/usr/bin/env python
# coding: utf-8
__doc__= '''check def main() for parameters and cli 
arguments/options
##### Copyright 2023
'''


import os
from six.moves import urllib

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
import pandas as pd
import seaborn as sns; 
# sns.set_context('notebook')
import tensorflow_datasets as tfds

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


import pickle
import sys
import numpy as np

from dist_dummy import DummyDistribution

# We will also do a quick check for availablility of a GPU:

# In[ ]:


if tf.test.gpu_device_name() != '/device:GPU:0':
  print("We'll just use the CPU for this run.")
else:
  print('Huzzah! Found GPU: {}'.format(tf.test.gpu_device_name()))


# ### Obtain Dataset:
# 
# We load the dataset from TensorFlow datasets and do some light preprocessing.

# In[ ]:


def load_data():
    fn = 'data.csv'
    df = pd.read_csv(fn)

    print(df.iloc[0])
    print(df.shape)
    features = df.iloc[:,2:13].values.astype('float32')
    # df['orgcode'] = df.county.astype(pd.api.types.CategoricalDtype())
    orgcode = df['OrgCode']
    # orgcode = df.orgcode
    count = df['Event'].astype(int)

    # import pdb;pdb.set_trace()
    return (features,orgcode,count),df



EPS = 1E-5

# ### Specify Model


# Initialize locations and scales randomly with `tf.Variable`s and 
# `tfp.util.TransformedVariable`s.
_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())

### use very wide uniform distribution for uninformed prior
# PRIOR_DIST = DummyDistribution
def _get_prior(val=None,val2=0,name=None):
  '''
  global trigger on how to deal with prior
  '''
  # _distc = _get_prior
  # _distc = tfd.Uniform

  ### Use a dummyDistribution to disable prior 
  _distc = DummyDistribution


  if val is None or isinstance(val,(float,int)):
    return _distc(-INF,INF,name=name)
  else:
    zval = tf.zeros_like(val)
    return _distc(-INF+zval,INF+zval,name=name)    

def make_joint_distribution_coroutine(feats, orgcode, norg, nfeat, method ='poisson'):
  '''
  For each method, initialise 3 objects
  '''
  nsample = feats.shape[0]
  if method == 'poisson':

    def model():
      ### Simple Poisson Model
      ### use very wide uniform distribution for uninformed prior
      # intercept      = yield _get_prior(-INF,INF,name='lograte_intercept')
      intercept      =    yield _get_prior(name='lograte_intercept')
      feat_rate_effect   = yield _get_prior(tf.zeros((nfeat,1)),name='feat_rate_effect')
      # feat_rate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield _get_prior(tf.zeros(norg),name='org_random_effect')
      
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 
      # log_rate_noise =  yield tfd.Normal(loc=log_rate, scale=1., name='log_rate_noise')


      yield tfd.Poisson(log_rate = log_rate,name='likelihood')

    def initer():
      return dict(
            lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect  = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            org_random_effect = tfd.Normal(_init_loc([norg]),_init_scale([norg]) ),            
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()

  elif method == 'zipoisson_shared_zero_rate':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # feat_rate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')


      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')

      log_zero_rate      = yield _get_prior(name='log_zero_rate')
      feat_rate_effect   = yield _get_prior(tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect  = yield _get_prior(tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      # log_zero_rate = yield _get_prior(-INF,INF,name='log_zero_rate')
      zero_prob = tf.sigmoid(log_zero_rate) + tf.zeros(nsample)

      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([zero_prob, 1.0 - zero_prob],axis=-1)),
          components=[tfd.Deterministic(loc=0+ tf.zeros(nsample)), tfd.Poisson(log_rate=log_rate)],
          name='likelihood'
      )      
      yield zero_inflated_poisson

    def initer():
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            # lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect  = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            org_random_effect = tfd.Normal(_init_loc([norg]),_init_scale([norg]) ),    
            log_zero_rate     = tfd.Normal(_init_loc(), _init_scale()),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      k = 'log_zero_rate'
      print(f'{k}  mean:{xd[k].mean()}  stddev:{xd[k].stddev()} ')


  elif method == 'zipoisson_zero_rate_by_org':

    def model():

        
      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept         = yield _get_prior(None,name='lograte_intercept')
      feat_rate_effect  = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      log_zero_rate = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='log_zero_rate')
      zero_prob =tf.sigmoid(tf.gather(log_zero_rate,orgcode,-1))

      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([zero_prob, 1.0 - zero_prob],axis=-1)),
          components=[tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=log_rate)],
          name='likelihood'
      )      
      yield zero_inflated_poisson

    def initer():
      
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            org_random_effect = tfd.Normal(_init_loc([norg]),_init_scale([norg]) ),    
            log_zero_rate = tfd.Normal(_init_loc([norg]), tf.clip_by_value(_init_scale([norg]),0,2) ),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      k = 'log_zero_rate'
      nmax = 5
      # xp 
      nh = 15
      xp = xd[k]
      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        header = f'{k}.{xpi}'
        header = header[:nh] + max(0,15-len(header))*' '+' '
        print(f'{header}    mean:{xpmean}   stddev:{xpdev}')
        if xpi==nmax:break

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
        if xpi==nmax:break
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')


      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')

  elif method == 'zipoisson_zero_rate_full':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      =    yield _get_prior(-INF,INF,name='lograte_intercept')
      feat_rate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      # log_zero_rate = yield tfd.Normal(tf.zeros(norg), scale = INF*tf.ones(norg), name='log_zero_rate')
      feat_zerorate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_zerorate_effect')

      log_zero_rate_org    = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='log_zero_rate') 
      zero_prob            = tf.sigmoid(tf.gather(log_zero_rate_org,orgcode,-1) + tf.squeeze(tf.matmul(feats,feat_zerorate_effect),-1))

      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([EPS+zero_prob, EPS+1.0 - zero_prob],axis=-1)),
          components=[tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=log_rate)],
          name='likelihood'
      )      
      yield zero_inflated_poisson

    def initer():
      
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            feat_zerorate_effect = tfd.Normal(_init_loc([nfeat,1]), tf.clip_by_value(_init_scale([nfeat,1]),0,2)) ,            
            org_random_effect = tfd.Normal(_init_loc([norg]), _init_scale([norg]) ),    
            log_zero_rate = tfd.Normal(_init_loc([norg]), tf.clip_by_value(_init_scale([norg]),0,2) ),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      k = 'log_zero_rate'
      nmax = 5
      # xp 
      nh = 15
      xp = xd[k]
      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        header = f'{k}.{xpi}'
        header = header[:nh] + max(0,15-len(header))*' '+' '
        print(f'{header}    mean:{xpmean}   stddev:{xpdev}')
        if xpi==nmax:break

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')

        if xpi==nmax:break

      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')

  elif method == 'zipoisson_orgcode_only':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield _get_prior(-INF,INF,name='lograte_intercept')
      

      feat_rate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)

      log_rate = tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) + intercept
      zero_prob = tf.sigmoid(org_random_effect_ins)
      

      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([EPS+zero_prob, EPS+1.0 - zero_prob],axis=-1)),
          components=[tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=log_rate)],
          name='likelihood'
      )      
      yield zero_inflated_poisson

    def initer():
      
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect = tfd.Normal(_init_loc([nfeat,1]), tf.clip_by_value(_init_scale([nfeat,1]),0,5) ),            
            # feat_zerorate_effect = tfd.Normal(_init_loc([nfeat,1]), tf.clip_by_value(_init_scale([nfeat,1]),0,2)) ,            
            org_random_effect = tfd.Normal(_init_loc([norg]),  tf.clip_by_value(_init_scale([norg]),0,5) ),    
            # log_zero_rate = tfd.Normal(_init_loc([norg]), tf.clip_by_value(_init_scale([norg]),0,2) ),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      # k = 'log_zero_rate'
      k = 'org_random_effect'
      nmax = 5
      # xp 
      nh = 15
      xp = xd[k]
      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        header = f'{k}.{xpi}'
        header = header[:nh] + max(0,15-len(header))*' '+' '
        print(f'{header}    mean:{xpmean}   stddev:{xpdev}')
        if xpi==nmax:break

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
        if xpi==nmax:break
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')

      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')      
  elif method == 'zipoisson_zero_rate_feat_only':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept          = yield _get_prior(-INF,INF,name='log_zerorate_intercept')
      feat_rate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect  = yield _get_prior(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      # log_zero_rate = yield tfd.Normal(tf.zeros(norg), scale = INF*tf.ones(norg), name='log_zero_rate')
      feat_zerorate_effect   = yield _get_prior(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_zerorate_effect')

      zero_prob = tf.sigmoid( tf.squeeze(tf.matmul(feats,feat_zerorate_effect),-1) + intercept)


      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([EPS+zero_prob, EPS+1.0 - zero_prob],axis=-1)),
          components=[tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=log_rate)],
          name='likelihood'
      )      
      yield zero_inflated_poisson

    def initer():
      
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            log_zerorate_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            feat_zerorate_effect = tfd.Normal(_init_loc([nfeat,1]), tf.clip_by_value(_init_scale([nfeat,1]),0,2)) ,            
            org_random_effect = tfd.Normal(_init_loc([norg]), _init_scale([norg]) ),    
            # log_zero_rate = tfd.Normal(_init_loc([norg]), tf.clip_by_value(_init_scale([norg]),0,2) ),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      # k = 'log_zero_rate'

  else:
    raise RuntimeError(f'Method {method!r} not understood!')    

  return tfd.JointDistributionCoroutineAutoBatched(model),initer, get_summary


INF = 10000000

def main():
  argv = sys.argv
  v = 1
  key = '--plot_png'
  if key in argv:
    v = argv[argv.index(key)+1]
    v = int(v)
  is_plot_png = v
  if is_plot_png:
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt

  ### gradient step size
  # learning_rate = 1e-2
  learning_rate = 5e-3

  ### whether to stop on nan
  test_nan = 0

  ### number of gradient steps
  num_steps = 3000
  # num_steps = 3000

  ### step interval to update visdom plots
  v = 100

  key = '--int_plot_vis'
  if key in argv:
    v = argv[argv.index(key)+1]
    v = int(v)
  int_plot_vis = v
  ### sample taken to estimate ELBO and gradient
  # sample_size=600
  # sample_size=2
  # sample_size=10
  sample_size=100
  
  ### load data
  (feats, orgcode, count),df = load_data()
  # df.head()
  
  ### whether to force recomputation 
  force = '--force' in argv

  ### set model recipe. see README.md for explanation
  v = 'poisson'
  # v = 'zipoisson_shared_zero_rate'
  # v = 'zipoisson_zero_rate_by_org'
  # v = 'zipoisson_zero_rate_full'
  # v = 'zipoisson_orgcode_only'
  v = 'zipoisson_zero_rate_feat_only'
  key = '--method'
  v = 'poisson'
  if key in argv:
    v = argv[argv.index(key)+1]
  method = v

  
  output_file = f'{method}.{num_steps}.pkl'


  def plot_vis(xdist,loss,output_file=output_file):
    if int_plot_vis>0:
      from visdom import Visdom
      vis = Visdom()

      key = 'org_random_effect'
      vis.boxplot(xdist[key].sample(1000).numpy(),win=output_file+'.'+key, opts={
        'title':f'{output_file}<br>{key}<br>loss:{loss}',
        'layoutopts': {
        'plotly': {
          'yaxis': {
            'type': 'linear',
            'range': [-2, 2],
            'autorange': False,
          }
        }
      }}
      
      )


      key = 'feat_rate_effect'
      # vis.line(xdist['feat_rate_effect'].mean().numpy(),win=output_file, opts={'title':output_file,'ymin':-5,'ymax':5})
      vis.boxplot(xdist[key].sample(1000).numpy(),win=output_file+'.'+key, opts={
        'title':f'{output_file}<br>{key}<br>loss:{loss}',

        'layoutopts': {
        'plotly': {
          'yaxis': {
            'type': 'linear',
            'range': [-2, 2],
            'autorange': False,
          }
        }
      }}
      
      )  

  features = feats
  ### (nsample,nfeat)
  
  data_labels = count
  ### (nsample,)
  nfeat = features.shape[1]
  cts, edgs = np.histogram(count, np.arange(0,35))
  # n_cat = orgco
  norg = len(set(orgcode))
  if int_plot_vis>0:
    #### this is for plotting 

    from visdom import Visdom
    vis = Visdom()
    # vis.bar(edgs[:-1]+0.5, cts )
    vis.bar( cts,win='data_count_histogram')

  print(f'''
### Loading Data
Feature shape: {features.shape}
Labels shape:  {data_labels.shape}
  ''')

  if os.path.exists(output_file) and not force:
    ### skip computation if file already exists
    with open(output_file,'rb')  as f:
      lossvals, xpar, xdist = pickle.load(f)
  else:

    ### initialise instance
    joint, param_initer, get_summary = make_joint_distribution_coroutine(
        features,orgcode,norg,nfeat,
        # features.floor.values, features.county_code.values, df.county.nunique(),
        # df.floor.nunique()
        # method ='poisson',
        method = method,
        )


    # Define a closure over the joint distribution 
    # to condition on the observed labels.
    def target_log_prob_fn(**args):
      return joint.log_prob(**args, likelihood=data_labels)


    ### initialise parameter and thus surrogate_posterior
    init_param = param_initer()
    post = surrogate_posterior = tfd.JointDistributionNamedAutoBatched(init_param)
    tf.keras.callbacks.TerminateOnNaN().set_params(post.parameters)

    ### RMSprop is very stable

    # optimizer = tf.optimizers.Adam(learning_rate=1e-2)
    optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = tf.optimizers.RMSprop(learning_rate=5e-3)

    def callback(qty, post=post):
      print(f'[iter]{qty.step},  loss:{qty.loss:.1f}')
      if int_plot_vis>0 and ((qty.step)%int_plot_vis)==0:
        plot_vis(post.sample_distributions()[0], qty.loss)

      if not test_nan:
        return [qty.loss][-1]
      else:
        for ig,g in enumerate(qty.gradients):
          print(f'{ig} {tf.reduce_max(tf.math.abs(g)).numpy():.3f}')
        for p in qty.parameters:
          if tf.reduce_any(tf.math.is_nan(p)).numpy():

            print(p)
            print('NaN detected!')
            breakpoint()
        return qty.loss

    ### print some summary 
    get_summary(surrogate_posterior)

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn, 
        surrogate_posterior,
        optimizer=optimizer,
        # num_steps=3000, 
        # num_steps=300, 
        # trace_fn= lambda qty:[print(f'[iter]{qty.step},  loss:{qty.loss:.1f}'),qty.loss][-1],
        trace_fn = callback,
        num_steps=num_steps, 
        seed=42,
        sample_size=sample_size)

    lossvals = losses.numpy()
    xd,xsample = post.sample_distributions()

    print(f'Final Likelihood: {lossvals[-1]}')
    get_summary(post)
    with open(output_file,'wb') as f: 
        pickle.dump([lossvals,post.parameters, post.sample_distributions()[0]] ,f)

    xdist = xd
    xpar = post.parameters
    
    

  for lv in lossvals[-5:]:
    print('%.3f'%lv)
  if int_plot_vis>0:
    plot_vis(xdist,lossvals[-1])
  xdir = output_file+'.paramdir'
  if not os.path.exists(xdir): os.makedirs(xdir)

  singulars= {}
  for k,xdistt in xdist.items():
    # breakpoint()
    x = xdistt.mean().numpy()
    xdev = xdistt.stddev().numpy()
    shapel = xdistt.batch_shape.__len__()
    if shapel==0:
      singulars[k]= {}
      singulars[k]['mean'] = x
      singulars[k]['stddev'] = xdev
    elif shapel == 1:
      v = np.stack([x,xdev],-1)
      pd.DataFrame(v,columns='mean stddev'.split()).to_csv( os.path.join(xdir,k)+'.csv') 
    elif shapel == 2:
      v = np.concatenate([x,xdev],-1)
      pd.DataFrame(v,columns='mean stddev'.split()).to_csv( os.path.join(xdir,k)+'.csv') 
    if is_plot_png and k in 'feat_rate_effect':
      # matplotlib
            
      v3 = np.abs(v[:,0])+1.96*np.abs(v[:,1])
      rg = max(v3) * 1.05  ### upper and lower size
      title = 'Title: Fixed Effect for Features \n Parameter: '+k 
      xlab = 'Index'
      ylab = 'Value'
      DPI = 80
      FIG_H_PIXEL = 800
      FIG_W_PIXEL = 600
      dpi = DPI
      fig,axs = plt.subplots(1,1,figsize=[FIG_H_PIXEL/dpi,FIG_W_PIXEL/dpi],dpi=dpi)
      # plt.barh( range(len(v)), v[:,0], xerr=1.96 * v[:,1], align='center', alpha=0.5)
      # plt.errorbar( y=range(len(v)), x=v[:,0], yerr=0.25, xerr=1.96 * v[:,1], alpha=0.5,linestyle='')
      plt.errorbar( x=range(len(v)), y=v[:,0], xerr=0.25, yerr=1.96 * v[:,1], color='blue', alpha=0.5,linestyle='',elinewidth=3)
      plt.xticks( range(len(v)), np.arange(len(v))+1)
      plt.ylim(-rg,rg)
      plt.axhline(0)
      plt.xlabel(xlab)
      plt.ylabel(ylab)
      plt.title(title)
      fig.savefig(xdir+'/'+k+'.png',dpi=dpi)
      plt.close(fig)
      # pd.DataFrame(xdev).to_csv( os.path.join(xdir,k)+'.stddev.csv') 
  if singulars:
    # df = pd.DataFrame( [pd.Series(v) for v in singulars.values()],columns=singulars.keys())
    k = 'singular'
    pd.DataFrame(singulars).T.to_csv( os.path.join(xdir,k)+'.csv') 






if __name__=='__main__':
  main()
  print('[done]')
  import pdb;pdb.set_trace()


