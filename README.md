tensorflow==2.11.0
tensorflow-probability==0.19.0


## Usage

```bash
python3 main_zip.py 
```


### models explained:

- poisson: use feature fixed effect and orgcode random effect to predict lograte of poisson, best loglikelihood
- zipoisson_shared_zero_rate: same as poisson, but with a ZIP with a fixed zero rate shared between samples
- zipoisson_zero_rate_feat_only: only use feature to predict zero rate for ZIP. 
- zipoisson_zero_rate_by_org: only use random per-orgcode zero rate for ZIP.

## Question:

Which way do you want this implemented?

