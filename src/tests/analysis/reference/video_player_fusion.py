"""Convert rigid tracking boxes to a consistently defined pot-centre candidate."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .extract_video_reference import DEFAULT_OUT


def fuse_player(rigid,colors):
    # The generic tracker can grow a box upwards into the torso. Estimate its
    # pot-centre offset from early strong colour segmentations, without labels.
    use=(colors.confidence>.8)&(colors.lost==0)&(rigid.time_s>7)&(rigid.time_s<20.5)&(rigid.score>.75)
    delta=np.hypot(colors.pixel_x-rigid.pixel_x,colors.pixel_y-rigid.pixel_y)
    use &=delta<85
    if int(use.sum())<20:raise RuntimeError('Too few independent colour observations to calibrate the pot centre')
    corrected=[];backward=[];calibrations=[]
    for coord,dim,size in [('x','width',73.),('y','height',84.)]:
        x=(rigid.loc[use,dim]-size).to_numpy(float)
        y=(colors.loc[use,f'pixel_{coord}']-rigid.loc[use,f'pixel_{coord}']).to_numpy(float)
        opt=least_squares(lambda p:p[0]*x+p[1]-y,[0.,0.],loss='soft_l1',f_scale=5.,bounds=([-1.,-25.],[1.,25.]))
        correction=opt.x[0]*(rigid[dim]-size)+opt.x[1]
        corrected.append(rigid[f'pixel_{coord}']+correction)
        backdim=rigid.get(f'backward_{dim}',rigid[dim])
        backward.append(rigid[f'backward_{coord}']+opt.x[0]*(backdim-size)+opt.x[1])
        residual=np.abs(opt.fun)
        calibrations.append(dict(axis=coord,slope=float(opt.x[0]),offset=float(opt.x[1]),
                                 examples=int(use.sum()),median_residual_px=float(np.median(residual))))
    fwd=np.column_stack(corrected);back=np.column_stack(backward)
    fb=np.linalg.norm(fwd-back,axis=1)
    size_ok=(rigid.width.between(35,130)&rigid.height.between(35,170))
    both=size_ok&(rigid.score>=.78)&(rigid.backward_score>=.78)&(fb<=22)
    colour=np.column_stack([colors.pixel_x,colors.pixel_y])
    distance=np.linalg.norm(fwd-colour,axis=1)
    colour_agree=(colors.confidence>=.65)&(colors.lost==0)&size_ok&(rigid.score>=.75)&(distance<=25)
    valid=both|colour_agree
    xy=(fwd+back)/2
    xy[colour_agree]=colour[colour_agree]
    xy[~valid]=np.nan
    result=pd.DataFrame(index=rigid.index,data=dict(pixel_x=xy[:,0],pixel_y=xy[:,1],valid=valid,
                     score=np.minimum(rigid.score,rigid.backward_score),
                     forward_backward_error_px=fb,colour_agreement=colour_agree,
                     source=np.where(colour_agree,'rigid_and_colour','rigid_bidirectional')))
    return result,calibrations


def run():
    rigid=pd.read_csv(DEFAULT_OUT/'player_rigid.csv').set_index('frame')
    colors=pd.read_csv(DEFAULT_OUT/'player_color.csv').set_index('frame')
    result,fit=fuse_player(rigid,colors)
    result.to_csv(DEFAULT_OUT/'player_fused.csv')
    (DEFAULT_OUT/'pot_centre_calibration.json').write_text(json.dumps(fit,indent=2))
    print(json.dumps(fit,indent=2),flush=True)
    print('fused detections',int(result.valid.sum()),'of',len(result),flush=True)
    print(result[['pixel_x','pixel_y','valid','forward_backward_error_px']].iloc[::60].round(1).to_string(),flush=True)


if __name__=='__main__':run()
