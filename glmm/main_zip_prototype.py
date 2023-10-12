#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Fitting Generalized Linear Mixed-effects Models Using Variational Inference
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/probability/examples/Linear_Mixed_Effects_Model_Variational_Inference"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Model_Variational_Inference.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Model_Variational_Inference.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Model_Variational_Inference.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>


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



def load_and_preprocess_radon_dataset(state='MN'):
  """Load the Radon dataset from TensorFlow Datasets and preprocess it.
  
  Following the examples in "Bayesian Data Analysis" (Gelman, 2007), we filter
  to Minnesota data and preprocess to obtain the following features:
  - `county`: Name of county in which the measurement was taken.
  - `floor`: Floor of house (0 for basement, 1 for first floor) on which the
    measurement was taken.

  The target variable is `log_radon`, the log of the Radon measurement in the
  house.
  """
  ds = tfds.load('radon', split='train')
  radon_data = tfds.as_dataframe(ds)
  radon_data.rename(lambda s: s[9:] if s.startswith('feat') else s, axis=1, inplace=True)
  df = radon_data[radon_data.state==state.encode()].copy()

  df['radon'] = df.activity.apply(lambda x: x if x > 0. else 0.1)
  # Make county names look nice. 
  df['county'] = df.county.apply(lambda s: s.decode()).str.strip().str.title()
  # Remap categories to start from 0 and end at max(category).
  df['county'] = df.county.astype(pd.api.types.CategoricalDtype())
  df['county_code'] = df.county.cat.codes
  # Radon levels are all positive, but log levels are unconstrained
  df['log_radon'] = df['radon'].apply(np.log)

  # Drop columns we won't use and tidy the index 
  columns_to_keep = ['log_radon', 'floor', 'county', 'county_code']
  df = df[columns_to_keep].reset_index(drop=True)
 
  return df








# ### Specify Model

# In[ ]:


INF = 100000
def make_joint_distribution_coroutine(floor, county, n_counties, n_floors):

  def model():
    ### use very wide uniform distribution for uninformed prior
    county_scale = yield tfd.HalfNormal(scale=1., name='scale_prior')
    # intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
    intercept    = yield tfd.Uniform(-INF,INF,name='intercept')
    floor_weight = yield tfd.Normal(loc=0., scale=1., name='floor_weight')
    county_prior = yield tfd.Normal(loc=tf.zeros(n_counties),
                                    scale=county_scale,
                                    name='county_prior')
    
  # zero_inflated_poisson = tfd.Mixture(
  #     cat=tfd.Categorical(probs=[zero_prob, 1.0 - zero_prob]),
  #     components=[tfd.Deterministic(loc=0.0), tfd.Poisson(log_rate=poisson_log_rate)],
  # )


    random_effect = tf.gather(county_prior, county, axis=-1)

    # fixed_effect = intercept + floor_weight * floor
    # linear_response = fixed_effect + random_effect
    linear_response = intercept + floor_weight*floor + random_effect
    yield tfd.Normal(loc=linear_response, scale=1., name='likelihood')
    
  return tfd.JointDistributionCoroutineAutoBatched(model)


def main():


  df = load_and_preprocess_radon_dataset()
  df.head()



  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
  df.groupby('floor')['log_radon'].plot(kind='density', ax=ax1);
  ax1.set_xlabel('Measured log(radon)')
  ax1.legend(title='Floor')

  df['floor'].value_counts().plot(kind='bar', ax=ax2)
  ax2.set_xlabel('Floor where radon was measured')
  ax2.set_ylabel('Count')
  fig.suptitle("Distribution of log radon and floors in the dataset");


  fig, ax = plt.subplots(figsize=(22, 5));
  county_freq = df['county'].value_counts()
  county_freq.plot(kind='bar', ax=ax)
  ax.set_xlabel('County')
  ax.set_ylabel('Number of readings');

  features = df[['county_code', 'floor']].astype(int)
  data_labels = df[['log_radon']].astype(np.float32).values.flatten()

  joint = make_joint_distribution_coroutine(
      features.floor.values, features.county_code.values, df.county.nunique(),
      df.floor.nunique())


  # Define a closure over the joint distribution 
  # to condition on the observed labels.
  def target_log_prob_fn(**args):
    return joint.log_prob(**args, likelihood=data_labels)



  n_counties = df.county.nunique()



  # ### Specify surrogate posterior

  # We now put together a surrogate family $q_{\lambda}$, where the parameters $\lambda$ are trainable. In this case, our family is independent multivariate normal distributions, one for each parameter, and $\lambda = \{(\mu_j, \sigma_j)\}$, where $j$ indexes the four parameters. 
  # 
  # The method we use to fit the surrogate family uses `tf.Variables`. We also use `tfp.util.TransformedVariable` along with `Softplus` to constrain the (trainable) scale parameters to be positive. Additionally, we apply `Softplus` to the entire `scale_prior`, which is a positive parameter.
  # 
  # We initialize these trainable variables with a bit of jitter to aid in optimization.

  surrogate_posterior = tfd.JointDistributionNamedAutoBatched(
    dict(
      scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
      intercept = tfd.Normal(_init_loc(), _init_scale()),                           
      floor_weight = tfd.Normal(_init_loc(), _init_scale()),                        
      county_prior= tfd.Normal(_init_loc([n_counties]), _init_scale([n_counties])))
      )  


  # Note that this cell can be replaced with [`tfp.experimental.vi.build_factored_surrogate_posterior`](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/vi/build_factored_surrogate_posterior?version=nightly), as in:
  # 
  # ```python
  # surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
  #   event_shape=joint.event_shape_tensor()[:-1],
  #   constraining_bijectors=[tfb.Softplus(), None, None, None])
  # ```

  # ### Results

  # Recall that our goal is to define a tractable parameterized family of distributions, and then select parameters so that we have a tractable distribution that is close to our target distribution. 
  # 
  # We have built the surrogate distribution above, and can use [`tfp.vi.fit_surrogate_posterior`](https://www.tensorflow.org/probability/api_docs/python/tfp/vi/fit_surrogate_posterior), which accepts an optimizer and a given number of steps to find the parameters for the surrogate model minimizing the negative ELBO (which corresonds to minimizing the Kullback-Liebler divergence between the surrogate and the target distribution). 
  # 
  # The return value is the negative ELBO at each step, and the distributions in `surrogate_posterior` will have been updated with the parameters found by the optimizer.

  # In[ ]:


  # optimizer = tf.optimizers.Adam(learning_rate=1e-2)
  optimizer = tf.optimizers.RMSprop(learning_rate=1e-2)

  def tracer(args):
    loss,grads,vars = args
    print(loss)
    # return loss
    return args
  losses = tfp.vi.fit_surrogate_posterior(
      target_log_prob_fn, 
      surrogate_posterior,
      optimizer=optimizer,
      # num_steps=3000, 
      # num_steps=300, 
      trace_fn= lambda qty:[print(f'[iter]{qty.step},  loss:{qty.loss:.1f}'),qty.loss][-1],
      num_steps=1000, 
      seed=42,
      sample_size=2)


  post = surrogate_posterior
  xd,xsample = surrogate_posterior.sample_distributions()


  print('        intercept (mean): ', xd['intercept'].mean().numpy())
  print('     floor_weight (mean): ', xd['floor_weight'].mean().numpy())
  print(' scale_prior (approx. mean): ', tf.reduce_mean(xd['scale_prior'].sample(10000)).numpy())

  print(pd.DataFrame(data=dict(intercept=[1.462, tf.reduce_mean(xd['intercept'].mean()).numpy()],
                              floor=[-0.693, tf.reduce_mean(xd['floor_weight'].mean()).numpy()],
                              scale=[0.3282, tf.reduce_mean(xd['scale_prior'].sample(10000)).numpy()]),
                    index=['lme4', 'vi']))





# Initialize locations and scales randomly with `tf.Variable`s and 
# `tfp.util.TransformedVariable`s.
_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())


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
