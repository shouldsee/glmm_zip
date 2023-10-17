
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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


def plot_scatter_2d(xs,ys,xbins=None,ybins=None):
    if xbins is None:

        xbins = 20
        xbins = np.linspace(min(xs),max(xs)+2,xbins)        
    if ybins is None:
        ybins = 20
        ybins = np.linspace(min(ys),max(ys),ybins)

    fig,axs = plt.subplots(1,2,figsize=[12,4])
    axi = -1
    axi+=1; plt.sca(axs[axi])
    plt.scatter(xs,ys,3,marker='x')
    (cts, xgd, ygd)  = np.histogram2d(xs,ys,bins=(xbins, ybins ) )

    zs = np.log2(1+cts)
    axi+=1; plt.sca(axs[axi])
    # plt.subplots()
    left,right = min(xgd),max(xgd)
    bottom,top = min(ygd),max(ygd)
    plt.imshow(zs.T,origin='lower',extent=(left,right, bottom, top),aspect='auto')
    # plt.pcolormesh(zs,origin='lower',)
    # plt.set_ylabel(xgd)
    plt.xticks(xgd[1:]-0.5);
    plt.yticks(ygd);
    plt.colorbar()

    return fig
    # plt.xticks(ygd[1:]);



#### model file recognise the 
x = '''
Region                  Type_1
Scheme                  Type_1
IsClinical                   1
Incident_Year             2011
Notification_Year         2015
Grouped Claim                1
PortalClaim                  0
Injury                 Type_55
Cause                  Type_37
Specialty              Type_34
Location                Type_3
Age at incident             69
IsPatientMale                0
Distance                   5.1
Est_Settlement_Year       2021
Est_Claim_Outcome            1
Est_Value                 4900
Settlement_Year           2021
Claim_Outcome                1
Value                     3500
'''
x = x.strip()
x = [xx.rsplit(None,1)[0] for xx in x.splitlines()]
DATA_INPUT_COLS = x

def tidy_data(fn, test_fn):
    '''
    Load csv and tidy up data
    assemble data into feature matrix

    separate data into train and test in ratio
    '''
    
    df = pd.read_csv(fn)
    df = df[DATA_INPUT_COLS]  
    for k in 'Settlement_Year Incident_Year Notification_Year Est_Settlement_Year'.split():
        df[k] = df[k].str.split('/').str.get(0).astype(int)
    df['Value'] = df.eval('Value * Claim_Outcome')  ### patch negative values

    df['Est_Value'] = df['Est_Claim_Outcome'] * df['Est_Value']
    # df['_Int_Incident_Year'] = 

    for k in 'Region Scheme Incident_Year Notification_Year Injury Cause Specialty Location'.split():
    # for k in 'Region Scheme Incident_Year Notification_Year Injury Cause Specialty Location'.split():
        df[k] = pd.Categorical(df[k])
        pass
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



_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-0.01, maxval=0.01))


def _init_scale(shape=()):
  xs =   tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())
  xs = tf.clip_by_value( xs , 0.01,3)
  return xs
# _init_scale = lambda shape=(): tf.clip_by_value( , 0,3)


class MyModel(tf.Module):
  def __init__(self, nfeat, nyear, cols):
    super().__init__()
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    # self.w = tf.Variable(5.0)
    # self.b = tf.Variable(0.0)
    self.trainable_weights = []
    _reg = lambda x:[self.trainable_weights.append(x),x][1]
    self.columns = cols

    self.prob_to_settle_bias    = _reg( _init_loc((1,nyear)) )
    self.prob_to_settle_bias_input    = _reg( _init_loc((1,nfeat)) )
    self.prob_to_settle_coef    = _reg( _init_loc([nfeat,nyear])    )        
    self.value_log_rate_bias          = _reg( _init_loc([1,]))
    self.value_log_rate_bias_input    = _reg( _init_loc([1,nfeat,]))
    self.value_log_rate_coef    = _reg( _init_loc([nfeat,1]))
    self.value_zero_prob_bias   = _reg( _init_loc([1,]))
    self.value_zero_prob_bias_input   = _reg( _init_loc([1,nfeat,]))
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
    value_log_rate  = value_log_rate_bias  + tf.squeeze(tf.matmul( feats + self.value_log_rate_bias_input , value_log_rate_coef),-1)  
    value_zero_prob = value_zero_prob_bias + tf.squeeze(tf.matmul( feats + self.value_zero_prob_bias_input, value_zero_prob_coef),-1)
    # value_zero_prob = tf.sigmoid( 0.01* value_zero_prob)
    value_zero_prob = tf.sigmoid( value_zero_prob)

    prob_to_settle = prob_to_settle_bias + tf.matmul( feats+ self.prob_to_settle_bias_input, prob_to_settle_coef)

    nsample = len(x)
    
    xp = tfd.JointDistributionNamed(dict(
      claim_value =  
      # tfd.Independent(
          tfd.Mixture(
            cat        =  tfd.Categorical(probs=tf.stack([EPS + value_zero_prob, EPS + 1.0 - value_zero_prob],axis=-1)),
            components = [tfd.Deterministic(loc= 0. + tf.zeros(nsample,dtype='float32')), tfd.Poisson(log_rate=value_log_rate)],            
          name='value_likelihood')
          # ,1)
      ,
      
      year = 
      # tfd.Independent(
        tfd.Categorical(
        probs = tf.nn.softmax( prob_to_settle, axis=-1) +EPS
        ,name = 'year_likelihood')
      # ,1)
      )
      )

    x2 = year = tfd.Independent(tfd.Categorical(probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1),name = 'year_likelihood'))
    x3 = year = tfd.Categorical(probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1),name = 'year_likelihood')
    # breakpoint()
    x2 = tfd.Independent(x3,1)
    return xp,x2, x3

  def log_prob(self, x, y):
    # year,x2,x3 = tfd.Independent(tfd.Categorical(probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1),name = 'year_likelihood')
    # year = tfd.Independent(tfd.Categorical(probs = tf.nn.softmax( EPS+prob_to_settle, axis=-1),name = 'year_likelihood')

    xp,x2,x3 = self.__call__(x)
    # xp = self.__call__(x)
    yy = y['year']
    yyy = tf.concat([yy,yy],1)
    # breakpoint()

    y = {k:tf.transpose(v) for k, v in y.items()}
    
    lp = xp.log_prob(y)
    lp = tf.squeeze(lp,0)

    # print()
    # breakpoint()
    return lp



def step_train(
  model,
  output_file = "test.pkl",

  nfeat = 50,
  nyear = 20,
  nsample = 3000,

  claim_value = None,
  year  = None,
  input_feats = None,
  
  is_fit=1, 
  force = False,
  learning_rate = 0.01,
  epochs = 2,
  batch_size = 20,):

  claim_value = claim_value.astype('float32')
  year = year.astype('int32')
  input_feats = input_feats.astype('float32')


  def loss_fn(dat):
    x,y = dat["input_feats"], dict(claim_value=dat["claim_value"],year=dat["year"])
    lp = model.log_prob(x, y)
    loss = tf.reduce_mean(-lp,0)
    return loss
  


  optimizer = tf.optimizers.RMSprop( learning_rate=learning_rate )


  # Prepare the training dataset.
  train_dataset = tf.data.Dataset.from_tensor_slices(
    dict(input_feats=input_feats, claim_value = claim_value, year=year))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

  # Prepare the validation dataset.
  # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  # val_dataset = val_dataset.batch(batch_size)


  for epoch in range(epochs):
      print("Start of epoch %d" % (epoch,))

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
              loss_value = loss_fn(dat)

          if is_fit:
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # breakpoint()
            # print(model.trainable_weights[0])

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

          if step % 10 == 0:
              print(
                  f" epoch={epoch}  step={step}  batch_train_loss={loss_value:.2f} "  
                  + 
                  f"learning_rate={learning_rate:.6f}"
              )
              # print("Seen so far: %s samples" % ((step + 1) * batch_size))  

  return model


def init_model(nfeat,nyear,input_model_file):
  model = MyModel(nfeat,nyear,DATA_INPUT_COLS)

  if os.path.exists(input_model_file):
    is_fit = 1
    with open(input_model_file, 'rb' ) as f:
      xs = pickle.load(f)
      # xs['weight']
      assert model.columns == xs['columns'],f"Model file incompatible with current DATA_INPUT_COLS={DATA_INPUT_COLS}"
      for i,x in enumerate(xs['weights']):
        model.trainable_weights[i].assign(x)
        #  = x[:] 
      # breakpoint()
    print(f'[loaded] from file {input_model_file!r}')
  else:
    is_fit = 1
  return model



def make_qc_plots(model,x, dat, config, prefix):
  
  input_feats = dat['input_feats']
  claim_value = dat['claim_value']
  year = dat['year']
  nyear = config ['nyear']
  
  # samples = model(input_feats)[0].sample(len(input_feats))
  samples = model(input_feats)[0].sample(1)
  # breakpoint()
  
  try:
    x.pop('EstClaimValue',)
    x.pop('EstYear',)
  except Exception as e:
    print(e)

  x['EstClaimValue'] = tf.reduce_mean(samples['claim_value'],0)
  x['EstYear'] = tf.reduce_mean(samples['year'],0)
  xs = x['EstClaimValue'].values

  xs = x['EstClaimValue'].values
  ys = claim_value.ravel()
  xs = np.log2(1+xs)
  ys = np.log2(1+ys)
  fig = plot_scatter_2d(xs,ys)
  plt.ylabel("predicted value")
  plt.xlabel("actual value")
  plt.title(f'QC: Claim Value\n  size N={len(xs)}')
  fn = prefix+".qc_claim_value.png"
  fig.savefig(fn)
  print(f'[debug] outputing plots at  {fn!r}')

  plt.close()

  xs = x['EstYear'].values
  ys = year.ravel()
  fig = plot_scatter_2d(xs,ys)
  END_YEAR = "SettlementYear"
  START_YEAR = "Notification_Year"
  plt.ylabel("predicted value")
  plt.xlabel("actual value")
  plt.title(f'QC: year-to-settle = ({END_YEAR} - {START_YEAR})\n size N={len(xs)}')
  fn = prefix+".qc_year.png"
  print(f'[debug] outputing plots at  {fn!r}')
  fig.savefig(fn)
  plt.close()


  # df2 = pd.DataFrame(Total="EstClaim")
# def plot_provision(df, maxim):
  # nyear 
  _claim = dat['claim_value']
  _year = year
  zs = np.tensordot( _claim.T,  np.eye(nyear+1)[_year],1).squeeze((0,1))
  actual_agg = zs
  nsample = 2000

  #### predict the output for all cases for 2000 times and calculate uncertaintiyt
  ### input_feats (ncase, nfeat)  -> samples['claim_value'] (ncase, 2000)  # sample 2000 predictions for claim_value 
  ###                             -> samples['year']        (ncase, 2000)  # sample 2000 predictions for year to settle
  samples = model(input_feats)[0].sample(nsample)  ### returns a dict of "year" "claim_value"

  _claim = samples['claim_value'].numpy()
  _year = samples['year'].numpy()
  # breakpoint()
  zs = np.matmul( _claim[:,None],  np.eye(nyear+1)[_year]).squeeze(1)
  pred_agg = zs

  plt.close()
  fig,axs = plt.subplots(1,2,figsize=[12,4])
  ax = axs[0]
  plt.sca(ax)
  ax.plot(actual_agg,'x--',label="train_data")
  _m  = pred_agg.mean(axis=0)
  _sd = pred_agg.std(axis=0)
  # plt.errorbar( x=range(len(_m)), y=actual_agg, xerr=1., yerr=0.01, color='red', alpha=0.5,linestyle='',elinewidth=3)
  plt.errorbar( x=range(len(_m)), y=_m, xerr=0.25, yerr=1.96 * _sd, color='blue', alpha=0.5,linestyle='',elinewidth=3,label="prediction")
  plt.legend()
  plt.ylabel("claim-value")
  plt.xlabel("year-to-settle")
  plt.title(f'QC: Aggregated claim value \n simulation size N={nsample}')
  fn = prefix+".qc_agg-sum.png"
  print(f'[debug] outputing plots at {fn!r}')
  fig.savefig(fn)
  plt.close()



# breakpoint()
def _get_input_feats(x):
  '''
  convert dataset to features
  '''
  out = []
  ks = []
  for k in x:
    v = x[k]
    if k.startswith('Est') or (k in "Settlement_Year Value Claim_Outcome".split()):
      continue
    elif v.dtype=="category":
      v = v.cat.codes
      vm = v.max()
      xs = np.eye(vm+1)[v]
    elif k in ("IsClinical,PortalClaim,Age at incident,IsPatientMale,Distance".split(",")+ ["Grouped Claim"]):
      xs = v.values[:,None]
    else:
      # if v 
      print(repr(k),v.dtype)
      breakpoint()

    ks.append((k,x[k].dtype.__repr__()))
    out.append(xs)

    pass
  # print(ks)
  return np.concatenate(out,1)


# import docopt_subcommands as dsub
from docopt import docopt 
def optclean(args):
    for k in list(args):
        # k = k.strip('-<>')
        args[k.strip("-<>")] = args.pop(k)
    return args

def main():
  # argv = sys.argv
  # args = docopt(DOC_TEMPLATE.format(program='pred_claim.py'), version='1.0',argv=argv)
  args = docopt(DOC_TEMPLATE.format(program='pred_claim.py'), version='1.0',argv=None)
  optclean(args)
  _main(**args)
  # dsub.main(program="pred_claim.py",
  #  doc_template=DOC_TEMPLATE,argv=argv,commands=commands)



def _main(

  input_model_file,
  is_fit,
  data_csv,
  learning_step_list,
  output_model_prefix,
  batch_size,
  version,
  help,
):
  is_fit = int(is_fit)
  batch_size = int(batch_size)
  def parse_lr_hist(learning_step_str):
    lr_list = []
    for v in learning_step_str.split('_'):
      epc,lr = v.split('x')
      epc = int(epc)
      lr = lr.replace('d','.')
      lr = float(lr)
      lr_list.append((epc,lr))
    return lr_list

  lr_list = parse_lr_hist(learning_step_list)
  
  if '--test' in sys.argv:

    nfeat = 50
    nyear = 20
    nsample = 3000

    np.random.seed(10)

    claim_value = np.random.random((nsample,1))*10
    np.random.seed(10)
    year  = (10*(np.random.random((nsample,1))))
    np.random.seed(10)
    input_feats = np.random.random((nsample,nfeat))
    nfeat = 50
    nyear = 20
    nsample = 3000
    print(claim_value[:5])
    step_train(
      output_file = "test2.pkl",

      nfeat = nfeat,
      nyear = nyear,
      nsample = nsample,

      claim_value = claim_value,
      year  = year,
      input_feats = input_feats,

      learning_rate = 0.0001,
      epochs = 2,
      batch_size = 20,
    )

  else:
    fn = data_csv
    train_hist_str = ""
    if input_model_file:
      try:
        sp = input_model_file.rsplit('.',2)[1]
        _ = parse_lr_hist(sp)        
        train_hist_str = sp 
      except Exception as e:
        print(f'[fail] to extract train_hist from name {input_model_file} error:{e!r}')
        

    # diff_epoch = sum(x[0] for x in lr_list)
    # total_epoch = init_epoch + diff_epoch
    if not output_model_prefix:
      if not input_model_file:
        output_model_prefix = fn 
      else:
        try:
          output_model_prefix = input_model_file.rsplit('.',2)[1]
        except Exception as e:
          output_model_prefix = input_model_file
        
    odir = os.path.dirname(os.path.realpath(output_model_prefix))
    if not os.path.exists(odir):
      os.makedirs(odir)

    new_hist_str = learning_step_list
    if train_hist_str:
      new_hist_str = train_hist_str + "_" +learning_step_list
    output_model_file = f'{output_model_prefix}.{new_hist_str}.pkl'
    print(f'[param] output_model_prefix={output_model_prefix}')
    print(f'[param] output_model_file={output_model_file}')
    

    #### output model pickle file

    (x,x1) , _ = tidy_data(fn,None)

    year = (x['Settlement_Year'] - x['Notification_Year'].astype(int)).values[:,None]
    claim_value = (x['Value']).values[:,None]    
    input_feats = _get_input_feats(x)
    dat = dict(year=year,claim_value=claim_value,input_feats=input_feats)

    nfeat = input_feats.shape[1]
    nyear = np.max(year) + 1
    nsample = input_feats.shape[0]

    model = init_model(nfeat,nyear,input_model_file)
    conf = dict(nyear=nyear,nfeat=nfeat)

    make_qc_plots(model, x, dat, conf, output_model_file+".before-fit")


    diff_epochs = 0
    for (epochs,learning_rate) in lr_list:
      diff_epochs += epochs
      model = step_train(
        model = model,
        # output_file = output_file,

        nfeat = nfeat,
        nyear = nyear,
        nsample = nsample,

        claim_value = claim_value,
        year  = year,
        input_feats = input_feats,

        learning_rate = learning_rate,

        is_fit     = is_fit,
        epochs     = epochs,
        batch_size = batch_size,
      )    
    if diff_epochs:

      with open(output_model_file,'wb') as f: 
          pickle.dump(dict(weights=model.trainable_weights,columns = model.columns) ,f)
      print(f'[saved] to file {output_model_file!r}')

      make_qc_plots(model, x, dat, conf, output_model_file+".after-fit")


  sys.exit(0)

DOC_TEMPLATE = f"""{{program}}

Usage: {{program}} [options]  

Options:
  -h --help                Show this screen.
  -v --version             Show the program version.
  --is_fit IS_FIT          An integer to indicate whether to fit model [default: 1].
  --data_csv DATA_CSV      CSV file to input as data [default: "./dataset.csv"].
  --input_model_file INPUT_MODEL_FILE 
                           .pkl file to input as model [default: ""].
  --output_model_prefix OUTPUT_MODEL_PREFIX 
                           prefix under which to save output model file [default: ""].
  --learning_step_list LEARNING_STEP_LIST 
                           Underscore delimited combinations of $EPOCHSx$LEARNING_RATE,
                           with "d" instead of "." 
                           for example "100x0d01_100x0d001" means 100 epochs at 0.01 learning rate, 
                           then 100 epochs at 0.001 learning rate.
                           [default: 0x1].
  --batch_size BATCH_SIZE  Batch size for making gradient step  [default: 30000].


Comment:
  Make sure DATA_CSV contain the following columns {DATA_INPUT_COLS!r}

Examples:
  ### initial fitting of the model
  python3 pred_claim.py --data_csv /tmp/dataset.csv --output_model_prefix ./mymodels/this-model --learning_step_list 100x0d01_100x0d001_100x0d0001 

  ### continue fitting the model
  python3 pred_claim.py --data_csv /tmp/dataset.csv --output_model_prefix ./mymodels/this-model --learning_step_list 100x0d0001  --input_model_file ./mymodels/this-model.100x0d01_100x0d001_100x0d0001.pkl

  ### plotting the first model without fitting
  python3 pred_claim.py --data_csv /tmp/dataset.csv --output_model_prefix ./test-plot --input_model_file ./mymodels/this-model.100x0d01_100x0d001_100x0d0001.pkl

  ### plotting the first on other datasets
  python3 pred_claim.py --data_csv other-dataset.csv --output_model_prefix ./test-plot --input_model_file ./mymodels/this-model.100x0d01_100x0d001_100x0d0001.pkl

"""



if __name__ == "__main__":
  main()