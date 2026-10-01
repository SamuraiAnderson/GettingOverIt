"""Fit a motion-based proposal for the player's screen position, not a detection."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from .extract_video_reference import DEFAULT_OUT


def run():
    flow=pd.read_csv(DEFAULT_OUT/'camera_translation.csv').set_index('frame')
    color=pd.read_csv(DEFAULT_OUT/'player_color.csv').set_index('frame')
    velocity=flow[['dx','dy']].interpolate(limit=3,limit_area='inside')
    velocity[:]=median_filter(velocity.to_numpy(),size=(3,1))
    base=(color.confidence>.65)&(color.lost==0)&(flow.time_s>10)&(flow.time_s<20.5)&(flow.flow_inliers>30)
    bests=[]
    for axis in range(2):
        y=color[['pixel_x','pixel_y']].iloc[:,axis]
        best=None
        for lag in range(-2,5):
            x=velocity.iloc[:,axis].shift(lag)
            use=base&x.notna()&y.notna()
            xx=x[use].to_numpy();yy=y[use].to_numpy()
            opt=least_squares(lambda p:p[0]*xx+p[1]-yy,[-7,640 if axis==0 else 400],loss='soft_l1',f_scale=10)
            errors=np.abs(opt.x[0]*xx+opt.x[1]-yy)
            score=np.median(errors)
            if best is None or score<best[0]:best=(score,lag,opt.x,errors)
        score,lag,p,errors=best
        print('axis',axis,'lag',lag,'model',p,'residual median / p90',score,np.percentile(errors,90),flush=True)
        flow['proposal_x' if axis==0 else 'proposal_y']=velocity.iloc[:,axis].shift(lag)*p[0]+p[1]
        bests.append(dict(axis=axis,lag=lag,slope=float(p[0]),intercept=float(p[1]),median_residual_px=float(score)))
    flow[['time_s','proposal_x','proposal_y']].to_csv(DEFAULT_OUT/'player_motion_proposals.csv')
    import json
    (DEFAULT_OUT/'camera_follow_fit.json').write_text(json.dumps(bests,indent=2))
    print(flow[['proposal_x','proposal_y']].iloc[::30].round(1).to_string())


if __name__=='__main__':run()
