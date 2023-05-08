#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2023


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




# ### Specify Model





# Initialize locations and scales randomly with `tf.Variable`s and 
# `tfp.util.TransformedVariable`s.
_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())
def make_joint_distribution_coroutine(feats, orgcode, norg, nfeat, method ='poisson'):
  nsample = feats.shape[0]
  if method == 'poisson':

    def model():
      ### Simple Poisson Model
      ### use very wide uniform distribution for uninformed prior
      intercept      = yield tfd.Uniform(-INF,INF,name='lograte_intercept')
      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

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

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield tfd.Uniform(-INF,INF,name='lograte_intercept')
      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      log_zero_rate = yield tfd.Uniform(-INF,INF,name='log_zero_rate')
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
            lograte_intercept = tfd.Normal(_init_loc(), _init_scale()),                           
            feat_rate_effect = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),            
            org_random_effect = tfd.Normal(_init_loc([norg]),_init_scale([norg]) ),    
            log_zero_rate = tfd.Normal(_init_loc(), _init_scale()),                           
                    
            # county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])),
            # log_zero_prob=  tfd.Normal(_init_loc(), _init_scale()),
            )
    def get_summary(post):
      xd,xsample = post.sample_distributions()
      k = 'log_zero_rate'
      print(f'{k}  mean:{xd[k].mean()}  stddev:{xd[k].stddev()} ')


  elif method == 'zipoisson_zero_rate_by_org':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield tfd.Uniform(-INF,INF,name='lograte_intercept')
      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      log_zero_rate = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='log_zero_rate')
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

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')


      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')

  elif method == 'zipoisson_zero_rate_full':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield tfd.Uniform(-INF,INF,name='lograte_intercept')
      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      # log_zero_rate = yield tfd.Normal(tf.zeros(norg), scale = INF*tf.ones(norg), name='log_zero_rate')
      feat_zerorate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_zerorate_effect')

      log_zero_rate_org = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='log_zero_rate') 
      zero_prob = tf.sigmoid(tf.gather(log_zero_rate_org,orgcode,-1) + tf.squeeze(tf.matmul(feats,feat_zerorate_effect),-1))

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

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')


      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')

  elif method == 'zipoisson_orgcode_only':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield tfd.Uniform(-INF,INF,name='lograte_intercept')
      

      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)

      log_rate = tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) + intercept
      zero_prob = tf.sigmoid(org_random_effect_ins)
      
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

      for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
        # xd[k]):
        print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
      # for xpi,xpp in enumerate(xd[k]):
      #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')


      # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')      
  elif method == 'zipoisson_zero_rate_feat_only':

    def model():
      ### use very wide uniform distribution for uninformed prior

      # county_scale   = yield tfd.HalfNormal(scale=1., name='scale_prior')
      # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
      intercept      = yield tfd.Uniform(-INF,INF,name='log_zerorate_intercept')
      feat_rate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_rate_effect')
      org_random_effect = yield tfd.Uniform(-INF+tf.zeros(norg),INF+tf.zeros(norg),name='org_random_effect')
      org_random_effect_ins = tf.gather( org_random_effect, orgcode,axis=-1)
      log_rate = org_random_effect_ins + tf.squeeze(tf.matmul( feats,feat_rate_effect),-1) 

      # log_zero_rate = yield tfd.Normal(tf.zeros(norg), scale = INF*tf.ones(norg), name='log_zero_rate')
      feat_zerorate_effect   = yield tfd.Uniform(-INF+tf.zeros((nfeat,1)),INF+tf.zeros((nfeat,1)),name='feat_zerorate_effect')

      zero_prob = tf.sigmoid( tf.squeeze(tf.matmul(feats,feat_zerorate_effect),-1) + intercept)

      zero_inflated_poisson = tfd.Mixture(
          cat=tfd.Categorical(probs=tf.stack([zero_prob, 1.0 - zero_prob],axis=-1)),
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
      
      # nmax = 5
      # # xp 
      # nh = 15
      # xp = xd[k]
      # for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
      #   # xd[k]):
      #   header = f'{k}.{xpi}'
      #   header = header[:nh] + max(0,15-len(header))*' '+' '
      #   print(f'{header}    mean:{xpmean}   stddev:{xpdev}')

      # for xpi,(xpmean,xpdev) in enumerate(zip(xp.mean(),xp.stddev())):
      #   # xd[k]):
      #   print(f'{k}.{xpi}.sigmoid  mean:{tf.sigmoid(xpmean)}   stddev:{xpdev}')
      # # for xpi,xpp in enumerate(xd[k]):
      # #   print(f'{k}.sigmoid.{xpi} {tf.sigmoid(xpp.mean())}   {(xpp.stddev())}')


      # # print(f'{k}          mean:{xd[k].mean()[:nmax]}  stddev:{xd[k].stddev()[:nmax]} ')
      # # print(f'{k}.sigmoid  mean:{tf.sigmoid(xd[k].mean()[:nmax])}  stddev:{xd[k].stddev()[:nmax]} ')

  else:
    raise RuntimeError('Method {method!r} not understood!')    

  return tfd.JointDistributionCoroutineAutoBatched(model),initer, get_summary


INF = 10000000
import pickle
import sys

import numpy as np
def main():
  # learning_rate = 1e-2
  learning_rate = 5e-3
  test_nan = 0
  num_steps = 3000
  # num_steps = 3000
  int_plot_vis = 100
  # sample_size=600
  sample_size=2
  (feats, orgcode, count),df = load_data()
  # df.head()
  method = 'poisson'
  # method = 'zipoisson_shared_zero_rate'
  # method = 'zipoisson_zero_rate_by_org'
  # # method = 'zipoisson_zero_rate_full'
  # method = 'zipoisson_orgcode_only'
  # method = 'zipoisson_zero_rate_feat_only'
  argv = sys.argv
  output_file = f'{method}.{num_steps}.pkl'
  force = '--force' in argv

  def plot_vis(xdist,loss,output_file=output_file):
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
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    # ax1.hist(count,pd.np.arange(0,30))
    # fig.show()


    from visdom import Visdom
    vis = Visdom()
    # vis.bar(edgs[:-1]+0.5, cts )
    vis.bar( cts )

  print(f'''
### Loading Data
Feature shape: {features.shape}
Labels shape:  {data_labels.shape}
  ''')

  if os.path.exists(output_file) and not force:
    with open(output_file,'rb')  as f:
      lossvals, xpar, xdist = pickle.load(f)
  else:



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



    # ### Specify surrogate posterior






    init_param = param_initer()
    post = surrogate_posterior = tfd.JointDistributionNamedAutoBatched(init_param)
    tf.keras.callbacks.TerminateOnNaN().set_params(post.parameters)

    # ### Results


    # optimizer = tf.optimizers.Adam(learning_rate=1e-2)
    optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = tf.optimizers.RMSprop(learning_rate=5e-3)

    def tracer(args):
      loss,grads,vars = args
      print(loss)
      # return loss
      return args
    post = surrogate_posterior

    get_summary(surrogate_posterior)
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

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn, 
        surrogate_posterior,
        optimizer=optimizer,
        # num_steps=3000, 
        # num_steps=300, 
        # trace_fn= lambda qty:[print(f'[iter]{qty.step},  loss:{qty.loss:.1f}'),qty.loss][-1],
        trace_fn = callback,
        # trace_fn= lambda qty:[print(f'[iter]{qty.step},  loss:{qty.loss:.1f}'),
        #   callback(qty,post),
        #   print(post.sample_distributions()[0]['log_zero_rate'].mean().numpy()[:]),
        #   print(post.sample_distributions()[0]['log_zero_rate'].stddev().numpy()[:]),
        #   qty.loss][-1],
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

  plot_vis(xdist,lossvals[-1])
    # plt.plot()






if __name__=='__main__':
  main()
  import pdb;pdb.set_trace()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(losses, 'k-')
ax.set(xlabel="Iteration",
       ylabel="Loss (ELBO)",
       title="Loss during training",
       ylim=0);



county_counts = (df.groupby(by=['county', 'county_code'], observed=True)
                   .agg('size')
                   .sort_values(ascending=False)
                   .reset_index(name='count'))

means = xd['county_weights_'].mean()
stds = xd['county_weights_'].stddev()

fig, ax = plt.subplots(figsize=(20, 5))

for idx, row in county_counts.iterrows():
  mid = means[row.county_code]
  std = stds[row.county_code]
  ax.vlines(idx, mid - std, mid + std, linewidth=3)
  ax.plot(idx, means[row.county_code], 'ko', mfc='w', mew=2, ms=7)

ax.set(
    xticks=np.arange(len(county_counts)),
    xlim=(-1, len(county_counts)),
    ylabel="County effect",
    title=r"Estimates of county effects on log radon levels. (mean $\pm$ 1 std. dev.)",
)
ax.set_xticklabels(county_counts.county, rotation=90);


# Indeed, we can see this more directly by plotting the log-number of observations against the estimated standard deviation, and see the relationship is approximately linear.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.log1p(county_counts['count']), stds.numpy()[county_counts.county_code], 'o')
ax.set(
    ylabel='Posterior std. deviation',
    xlabel='County log-count',
    title='Having more observations generally\nlowers estimation uncertainty'
);


# 
# ## Comparing to `lme4` in R

# In[ ]:


# get_ipython().run_cell_magic('shell', '', "exit  # Trick to make this block not execute.\n\nradon = read.csv('srrs2.dat', header = TRUE)\nradon = radon[radon$state=='MN',]\nradon$radon = ifelse(radon$activity==0., 0.1, radon$activity)\nradon$log_radon = log(radon$radon)\n\n# install.packages('lme4')\nlibrary(lme4)\nfit <- lmer(log_radon ~ 1 + floor + (1 | county), data=radon)\nfit\n\n# Linear mixed model fit by REML ['lmerMod']\n# Formula: log_radon ~ 1 + floor + (1 | county)\n#    Data: radon\n# REML criterion at convergence: 2171.305\n# Random effects:\n#  Groups   Name        Std.Dev.\n#  county   (Intercept) 0.3282\n#  Residual             0.7556\n# Number of obs: 919, groups:  county, 85\n# Fixed Effects:\n# (Intercept)        floor\n#       1.462       -0.693")


# The following table summarizes the results.

# In[ ]:



# This table indicates the VI results are within ~10% of `lme4`'s.  This is somewhat surprising since:
# - `lme4` is based on [Laplace's method](https://www.jstatsoft.org/article/view/v067i01/) (not VI),
# - no effort was made in this colab to actually converge,
# - minimal effort was made to tune hyperparameters,
# - no effort was taken regularize or preprocess the data (eg, center features, etc.).

# ## Conclusion

# In this colab we described Generalized Linear Mixed-effects Models and showed how to use variational inference to fit them using TensorFlow Probability. Although the toy problem only had a few hundred training samples, the techniques used here are identical to what is needed at scale.
