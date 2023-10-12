import os
from six.moves import urllib


import pandas as pd
def load_data():
    fn = 'data.csv'
    df = pd.read_csv(fn)

    print(df.iloc[0])
    print(df.shape)
    features = df.iloc[:,2:13]
    # df['orgcode'] = df.county.astype(pd.api.types.CategoricalDtype())
    orgcode = df['OrgCode']
    # orgcode = df.orgcode
    count = df['Event']

    # import pdb;pdb.set_trace()
    return (features,orgcode,count),df



def main():
    load_data()
    import pdb;pdb.set_trace()

def make_joint_distribution_coroutine(floor, county, n_counties, n_floors):

    def model():
        county_scale = yield tfd.HalfNormal(scale=1., name='scale_prior')
        intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
        floor_weight = yield tfd.Normal(loc=0., scale=1., name='floor_weight')
        county_prior = yield tfd.Normal(loc=tf.zeros(n_counties),
                                        scale=county_scale,
                                        name='county_prior')
        random_effect = tf.gather(county_prior, county, axis=-1)

        fixed_effect = intercept + floor_weight * floor
        linear_response = fixed_effect + random_effect
        yield tfd.Normal(loc=linear_response, scale=1., name='likelihood')
    return tfd.JointDistributionCoroutineAutoBatched(model)
    joint = make_joint_distribution_coroutine(
        features.floor.values, features.county_code.values, df.county.nunique(),
        df.floor.nunique())

    # Define a closure over the joint distribution 
    # to condition on the observed labels.
    def target_log_prob_fn(*args):
        return joint.log_prob(*args, likelihood=labels)


if __name__=='__main__':
    main()
