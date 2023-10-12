
import os
from random import sample
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


def tidy_data(fn, test_fn):
    '''
    Load csv and tidy up data
    assemble data into feature matrix

    separate data into train and test in ratio
    '''
    
    df = pd.read_csv(fn)
    for k in 'Settlement_Year Incident_Year Notification_Year Est_Settlement_Year'.split():
        df[k] = df[k].str.split('/').str.get(0).astype(int)
    df['Value'] = df.eval('Value * Claim_Outcome')  ### patch negative values

    df['Est_Value'] = df['Est_Claim_Outcome'] * df['Est_Value']


    train_input = df
    train_output= None
    test_input  = None
    test_output = None

    return (train_input, train_output),(test_input,test_output)        
    

### a very small number
EPS = 1E-5
### a very big number
INF = 10000000


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


# def train(model, data, output_fn):
# def make_joint_distribution_coroutine(feats, orgcode, norg, nfeat, method ='poisson'):

_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))
def _init_scale(shape=()):
  xs =   tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())
  xs = tf.clip_by_value( xs , 0.01,3)
  return xs
# _init_scale = lambda shape=(): tf.clip_by_value( , 0,3)

def make_joint_distribution_coroutine(
    # feats   = None,
    nyear = 10,
    nfeat = 50,
    method = None,
):


    def model(feats):
        ### use very wide uniform distribution for uninformed prior

        nsample = len(feats)


        prob_to_settle_bias  = yield _get_prior(-INF+tf.zeros((1,nyear)), INF+tf.zeros((1,nyear)),name='prob_to_settle_bias')
        prob_to_settle_coef  = yield _get_prior(-INF+tf.zeros((nfeat,nyear)), INF+tf.zeros((nfeat,nyear)),name='prob_to_settle_coef')
        value_log_rate_bias  = yield _get_prior(-INF,INF,name='value_log_rate_bias')
        value_log_rate_coef  = yield _get_prior(-INF+tf.zeros((nfeat,1)), INF+tf.zeros((nfeat,1)),name='value_log_rate_coef')
        value_zero_prob_bias = yield _get_prior(-INF,INF,name='value_zero_prob_bias')
        value_zero_prob_coef = yield _get_prior(-INF+tf.zeros((nfeat,1)), INF+tf.zeros((nfeat,1)),name='value_zero_prob_coef')


    #   value_log_rate  = yield _get_prior(-INF,INF,name='lograte_intercept')
        value_log_rate  = value_log_rate_bias  + tf.squeeze(tf.matmul( feats, value_log_rate_coef),-1)  
        value_zero_prob = value_zero_prob_bias + tf.squeeze(tf.matmul( feats, value_zero_prob_coef),-1)
        value_zero_prob = tf.sigmoid(value_zero_prob)

        prob_to_settle = prob_to_settle_bias + tf.matmul( 
            feats, prob_to_settle_coef)
        
        # # 
        # joint_likelihood = tfd.JointDistributionNamed(dict(
        #   value = tfd.Mixture(
        #       cat        =  tfd.Categorical(probs=tf.stack([value_zero_prob, 1.0 - value_zero_prob],axis=-1)),
        #       components = [tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=value_log_rate)],
        #   ),
        #   year_to_settle = tfd.Categorical(
        #       probs = tf.nn.softmax( prob_to_settle, axis=-1)
        #       )
        #   ),
        #   name='joint_likelihood'        
        # )

        #   xp = tfd.Mixture(
        #         cat        =  tfd.Categorical(probs=tf.stack([value_zero_prob, 1.0 - value_zero_prob],axis=-1)),
        #         components = [tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=value_log_rate)],
        #       )
            
        #   joint_likelihood = tfd.JointDistributionSequential([
        #       tfd.Mixture(
        #         cat        =  tfd.Categorical(probs=tf.stack([value_zero_prob, 1.0 - value_zero_prob],axis=-1)),
        #         components = [tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=value_log_rate)],
        #       ),
        #       tfd.Categorical(
        #         probs = tf.nn.softmax( prob_to_settle, axis=-1)
        #       )],)

        value_likelihood = yield tfd.Mixture(
            cat        =  tfd.Categorical(probs=tf.stack([EPS + value_zero_prob, EPS + 1.0 - value_zero_prob],axis=-1)),
            components = [tfd.Deterministic(loc= 0 + tf.zeros(nsample)), tfd.Poisson(log_rate=value_log_rate)],            
        name='value_likelihood')

        year_likelihood = yield tfd.Categorical(
            probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1)
            ,name = 'year_likelihood')
        # breakpoint()
        # yield tfd.JointDistributionSequential([value_likelihood, year_likelihood])
      
    def initer():
      
      return dict(
            # scale_prior=tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),           
            prob_to_settle_bias    = tfd.Normal(_init_loc((1,nyear)), _init_scale((1,nyear))),                           
            prob_to_settle_coef    = tfd.Normal(_init_loc([nfeat,nyear]), _init_scale([nfeat,nyear])),            
            value_log_rate_bias    = tfd.Normal(_init_loc([1,]), _init_scale([1,])),
            value_log_rate_coef    = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),
            value_zero_prob_bias   = tfd.Normal(_init_loc([1,]), _init_scale([1,])),
            value_zero_prob_coef   = tfd.Normal(_init_loc([nfeat,1]), _init_scale([nfeat,1])),
                    
            )

    def get_summary(post):
      xd,xsample = post.sample_distributions()
      k = 'value_log_rate_coef'
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
  
    return model,initer, get_summary




def step_train_model(
  output_dict = dict(value=None,year_to_settle=None),
  input_feats = None,
  nfeat = 50,
  nyear = 10,
  # is_test_nan = False,
  is_test_nan = True,
  output_file = "temp.pkl",
  force = False,
  sample_size = 20,
  num_steps =100,
  learning_rate = 0.01,
):



  if os.path.exists(output_file) and not force:
    ### skip computation if file already exists
    with open(output_file,'rb')  as f:
      lossvals, xpar, xdist = pickle.load(f)
  else:

    ### initialise instance
    model_binder, param_initer, get_summary = make_joint_distribution_coroutine(
        # feats   = None,
        nyear = nyear,
        nfeat = nfeat,
        method = None,
      )
    model = lambda :model_binder(input_feats)
    joint = tfd.JointDistributionCoroutineAutoBatched(model)

    def target_log_prob_fn(**args):
    #   return joint.log_prob(**args, joint_likelihood=output_dict)
      # return joint.log_prob(**args, value_likelihood=output_dict['value'], year_likelihood = output_dict['year'])
      # return 
      loss = joint.log_prob(**args, value_likelihood=output_dict['value'], year_likelihood = output_dict['year'])
      loss = tf.reduce_mean(loss, axis=0)
      print(f'[prob]{loss}')
      return loss


    ### initialise parameter and thus surrogate_posterior
    init_param = param_initer()
    post = surrogate_posterior = tfd.JointDistributionNamedAutoBatched(init_param)
    # post = surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(init_param)

    ### RMSprop is very stable
    # optimizer = tf.optimizers.Adam(learning_rate=1e-2)
    optimizer = tf.optimizers.RMSprop( learning_rate=learning_rate )

    def callback(qty, post=post):
      # breakpoint()
      print(f'[iter]{qty.step},  loss:{qty.loss:.1f}')

      if not is_test_nan:
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


class MyModel(tf.Module):
  def __init__(self, nfeat, nyear):
    super().__init__()
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    # self.w = tf.Variable(5.0)
    # self.b = tf.Variable(0.0)
    self.trainable_weights = []
    _reg = lambda x:[self.trainable_weights.append(x),x][1]

    self.prob_to_settle_bias    = _reg( _init_loc((1,nyear)) )
    self.prob_to_settle_coef    = _reg( _init_loc([nfeat,nyear])    )        
    self.value_log_rate_bias    = _reg( _init_loc([1,]))
    self.value_log_rate_coef    = _reg( _init_loc([nfeat,1]))
    self.value_zero_prob_bias   = _reg( _init_loc([1,]))
    self.value_zero_prob_coef   = _reg( _init_loc([nfeat,1]))


  def __call__(self, x):
    feats= x
    value_log_rate_bias= self.value_log_rate_bias
    value_zero_prob_bias = self.value_zero_prob_bias
    value_log_rate_coef = self.value_log_rate_coef
    value_zero_prob_coef = self.value_zero_prob_coef
    prob_to_settle_bias = self.prob_to_settle_bias
    prob_to_settle_coef = self.prob_to_settle_coef

#   value_log_rate  = yield _get_prior(-INF,INF,name='lograte_intercept')
    value_log_rate  = value_log_rate_bias  + tf.squeeze(tf.matmul( feats, value_log_rate_coef),-1)  
    value_zero_prob = value_zero_prob_bias + tf.squeeze(tf.matmul( feats, value_zero_prob_coef),-1)
    value_zero_prob = tf.sigmoid(value_zero_prob)

    prob_to_settle = prob_to_settle_bias + tf.matmul( 
        feats, prob_to_settle_coef)

    nsample = len(x)
    
    xp = tfd.JointDistributionNamed(dict(
      claim_value =  tfd.Independent(
          tfd.Mixture(
            cat        =  tfd.Categorical(probs=tf.stack([EPS + value_zero_prob, EPS + 1.0 - value_zero_prob],axis=-1)),
            components = [tfd.Deterministic(loc= 0. + tf.zeros(nsample,dtype='float32')), tfd.Poisson(log_rate=value_log_rate)],            
          name='value_likelihood'),1
      ),
      
      year = tfd.Independent(
        tfd.Categorical(
        probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1)
        ,name = 'year_likelihood')
      ,1)
      )
      )
    return xp

  def log_prob(self, x, y):
    xp = self.__call__(x)
    return xp.log_prob(y)





def main():
  if '--test' in sys.argv:


    nfeat = 50
    nyear = 20
    nsample = 3000

    value = np.random.random((nsample,1)).astype('float32')*10
    # value = value.astype( "int8" )
    year  = (10*(np.random.random((nsample,1)))).astype('int32')
    input_feats = np.random.random((nsample,nfeat)).astype('float32')


    model = MyModel(nfeat,nyear)
    def loss_fn(x,y):
      lp = model.log_prob(x, y)
      # print(lp.shape)
      loss = tf.reduce_mean(-lp,0)
      # print(loss.shape)
      # breakpoint()
      return loss


    learning_rate = 0.01


    epochs = 2
    batch_size = 20
    optimizer = tf.optimizers.RMSprop( learning_rate=learning_rate )
    x_train = input_feats
    y_train = tf.stack((value,year),axis=-1)


    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices(dict(input_feats=input_feats, claim_value = value, year=year))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # val_dataset = val_dataset.batch(batch_size)


    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (dat) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                # logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                # dat=attrdi
                loss_value = loss_fn(dat["input_feats"], dict(claim_value=dat["claim_value"],year=dat["year"]))

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))  
    sys.exit(0)
    tfd = tfp.distributions

    # Make independent distribution from a 2-batch Normal.
    ind = tfd.Blockwise(
        distributions=[
          tfd.Normal(loc=-1,scale=0.1),
          tfd.Normal(loc=-1,scale=0.1),
          tfd.Normal(loc=-1,scale=0.1),
          tfd.Normal(loc=-1,scale=0.1),
          ],
        # reinterpreted_batch_ndims=1
        )


    # tfd = tfp.distributions
    # joint = tfd.JointDistributionSequential([
    #                 tfd.Exponential(rate=[100, 120]),           # e
    #     lambda    e: tfd.Gamma(concentration=e[0], rate=e[1]),    # g
    #                 tfd.Normal(loc=0, scale=2.),                 # n
    #     lambda n, g: tfd.Normal(loc=n, scale=g),                  # m
    #     lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)      # x
    # ], batch_ndims=0, use_vectorized_map=True)
    joint = ind
    x = joint.sample(10)
    print(x.shape)
    breakpoint()

    nfeat = 50
    nyear = 20
    nsample = 300

    value = np.random.random((nsample,1)).astype('float32')*10
    value = value.astype( "int8" )
    year=(np.random.random((nsample,1))<0.5)
    
    step_train_model(
      output_dict = dict(value=value, year=year ) ,
      # output_dict = [np.random.random((nsample,1)).astype('float32'), np.random.random((nsample,1))],
      input_feats = np.random.random((nsample,nfeat)).astype('float32'),
      nfeat = nfeat,
      nyear = nyear,
      is_test_nan = False,
      output_file = "test.pkl",
      force = True,
      sample_size = 100,
      num_steps =300,
      learning_rate = 0.001,
    )
if __name__ == "__main__":
  main()