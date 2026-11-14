"""Training-side isotonic map fitting. Retained maps are loaded, not refitted, by default.

Callers must provide independent expert judgments from excluded training-side
calibration/OOF data. Never fit this function on test or outer-validation labels.
The optional weight-training runner intentionally retains the published maps.
"""
import numpy as np
def pava(scores,labels):
    x,inverse=np.unique(np.asarray(scores,np.float64),return_inverse=True)
    weights=np.bincount(inverse).astype(np.float64)
    sums=np.bincount(inverse,weights=np.asarray(labels,np.float64))
    blocks=[]
    for i in range(len(x)):
        blocks.append([i,i+1,weights[i],sums[i]])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            right,left=blocks.pop(),blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    y=np.empty_like(x)
    for lo,hi,w,s in blocks:y[lo:hi]=s/w
    return x,y

def fit_map(scores,labels):
    assert len(scores)==len(labels)>0 and np.isfinite(scores).all()
    x,y=pava(scores,labels)
    from sklearn.isotonic import IsotonicRegression
    check=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(scores,labels)
    assert np.allclose(np.interp(scores,x,y),check.predict(scores),atol=2e-14,rtol=0)
    return dict(x=list(map(float,x)),y=list(map(float,y)),judgments=len(scores),positives=int(sum(labels)))
