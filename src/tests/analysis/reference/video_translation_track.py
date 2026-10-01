import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .extract_video_reference import DEFAULT_OUT, load_map, project
from .video_edge_registration import EdgeMap


def repair_motion(df,max_missing=6):
    """Fill short losses in camera velocity, preserving raw optical-flow evidence."""
    out=df.copy()
    for axis in ['dx','dy']:
        if f'{axis}_raw' not in out:out[f'{axis}_raw']=out[axis]
    raw=out[['dx_raw','dy_raw']].to_numpy(float);values=raw.copy()
    valid=np.isfinite(raw).all(1);filled=np.zeros(len(out),bool)
    ids=np.flatnonzero(valid)
    for a,b in zip(ids[:-1],ids[1:]):
        if not 0<b-a-1<=max_missing:continue
        left,right=raw[a],raw[b]
        if np.dot(left,right)<0 or np.linalg.norm(left-right)>30:continue
        w=(np.arange(1,b-a)/(b-a))[:,None]
        values[a+1:b]=left*(1-w)+right*w;filled[a+1:b]=True
    out[['dx','dy']]=values;out['flow_interpolated']=filled
    displacement=np.nan_to_num(values)*np.array([-1,1])/72.
    out[['camera_x','camera_y']]=np.array([-42.16,-1.4])+np.cumsum(displacement,axis=0)
    return out


def translations(prev,curr):
    mask=np.full(prev.shape,255,np.uint8);mask[325:,:170]=0
    p0=cv2.goodFeaturesToTrack(prev,1600,.004,4,mask=mask,blockSize=5)
    if p0 is None or len(p0)<6:return []
    p1,st,_=cv2.calcOpticalFlowPyrLK(prev,curr,p0,None,winSize=(31,31),maxLevel=5)
    back,st2,_=cv2.calcOpticalFlowPyrLK(curr,prev,p1,None,winSize=(31,31),maxLevel=5)
    good=st.ravel().astype(bool)&st2.ravel().astype(bool)&(np.linalg.norm(back-p0,axis=2).ravel()<1.)
    a,b=p0[good,0],p1[good,0]
    delta=b-a
    if len(delta)<6:return []
    active=np.ones(len(delta),bool);groups=[]
    for _ in range(6):
        ids=np.where(active)[0]
        if len(ids)<6:break
        tree=cKDTree(delta[ids]);counts=tree.query_ball_point(delta[ids],.75,return_length=True)
        peak=ids[np.argmax(counts)];sel=active&(np.linalg.norm(delta-delta[peak],axis=1)<1.)
        active[sel]=False
        if sel.sum()<6:continue
        d=np.median(delta[sel],axis=0)*2
        spread=np.ptp(a[sel],axis=0)*2
        groups.append(dict(delta=d,n=int(sel.sum()),spread=spread))
    return groups


def run():
    _,poly,boundary=load_map();atlas=EdgeMap(poly,boundary,scale=72,down=4)
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,203)
    pose=np.array([72.,-42.16,-1.4]);prev=None;rows=[];lastshift=np.zeros(2)
    for fid in range(203,1304):
        ok,im=cap.read()
        if not ok:break
        gray=cv2.cvtColor(cv2.resize(im,(640,360)),cv2.COLOR_BGR2GRAY)
        best=None;groups=[]
        if prev is not None:
            groups=translations(prev,gray)
            candidates=[g for g in groups if g['n']>=16 and g['spread'].max()>170 and g['spread'].min()>35]
            if candidates:
                # Foreground is the fastest broad, coherently translating layer.
                best=max(candidates,key=lambda g:np.linalg.norm(g['delta']))
                d=best['delta'];pose[1:]+=np.array([-d[0],d[1]])/pose[0]
                lastshift=d
        error=np.nan;inliers=np.nan
        if (fid-203)%15==0:
            p,q=atlas.refine(im,pose)
            error=q['error_px'];inliers=q['inlier_fraction']
        rows.append(dict(frame=fid,time_s=fid/fps,scale=pose[0],camera_x=pose[1],camera_y=pose[2],
                         flow_inliers=best['n'] if best else 0,edge_error=error,edge_inliers=inliers,
                         dx=best['delta'][0] if best else np.nan,dy=best['delta'][1] if best else np.nan))
        if (fid-203)%30==0:print(fid,'pose',pose.round(2),'edge',round(error,2),round(inliers,2),
                                 'groups',[(g['n'],g['delta'].round(1).tolist()) for g in groups[:4]],flush=True)
        prev=gray
    cap.release();repair_motion(pd.DataFrame(rows)).to_csv(DEFAULT_OUT/'camera_translation.csv',index=False)


if __name__=='__main__':
    import sys
    if '--repair-existing' in sys.argv:
        result=repair_motion(pd.read_csv(DEFAULT_OUT/'camera_translation.csv'))
        result.to_csv(DEFAULT_OUT/'camera_translation.csv',index=False)
        print('interpolated camera velocities',int(result.flow_interpolated.sum()),flush=True)
    else:run()
