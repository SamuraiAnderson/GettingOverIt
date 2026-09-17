"""Re-anchor motion estimates on the terrain map and leave unsupported frames missing."""
import cv2
import json
import numpy as np
import pandas as pd
from extract_video_reference import DEFAULT_OUT,load_map,project,frame_edges
from video_edge_registration import EdgeMap


def run():
    _,poly,boundary=load_map();atlas=EdgeMap(poly,boundary,scale=72,down=4)
    motion=pd.read_csv(DEFAULT_OUT/'camera_translation.csv').set_index('frame')
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'))
    initial_pose=np.asarray(json.loads((DEFAULT_OUT/'initial_pose.json').read_text())['pose'])
    global_rows=json.loads((DEFAULT_OUT/'global_candidates.json').read_text())
    global_anchors=[]
    for row in global_rows:
        for rank,candidate in enumerate(row['candidates']):
            if candidate['error_px']<4 and candidate['inlier_fraction']>.60 and candidate['n_boundary']>100:
                raw=motion.loc[row['frame'],['camera_x','camera_y']].to_numpy(float)
                p=np.asarray(candidate['pose'])
                # Require a global candidate to agree with the independent motion route.
                alternative=row['candidates'][1]['error_px'] if len(row['candidates'])>1 else 30.
                distinctive_late=(row['time_s']>33 and rank==0 and alternative-candidate['error_px']>1.5)
                # A distinct whole-map match may re-seed after sparse dark-scene
                # optical flow. It is only a proposal; local gates still apply.
                radius=45 if distinctive_late else 16
                if np.linalg.norm(p[1:]-raw)<radius:
                    global_anchors.append((row['frame'],p[1:]-raw))
                    break
    offset=np.zeros(2);rows=[];failures=0
    for fid in range(203,1304,9):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read()
        if not ok:break
        raw=motion.loc[fid,['camera_x','camera_y']].to_numpy(float)
        if global_anchors:
            anchor=min(global_anchors,key=lambda a:abs(a[0]-fid))
            if abs(anchor[0]-fid)<60 and (failures>=3 or abs(anchor[0]-fid)<18):offset=anchor[1]
        prediction=raw+offset
        radius=(3,3) if failures<4 else (7,14)
        candidates=atlas.search(im,prediction,radius,top=8,prior_weight=.3 if failures<4 else .03,min_edge_pixels=75)
        candidates.append((0.,float(prediction[0]),float(prediction[1]),0))
        fits=[]
        for cost,x,y,n in candidates:
            pose,q=atlas.refine(im,[72,x,y])
            uv=project(boundary,pose)
            use=(uv[:,0]>12)&(uv[:,0]<1268)&(uv[:,1]>12)&(uv[:,1]<650)
            xy=uv[use]
            spread=np.ptp(xy,axis=0) if len(xy) else np.zeros(2)
            # Small snippets and straight lines do not determine two-dimensional position.
            eigen=np.linalg.eigvalsh(np.cov(xy.T)) if len(xy)>2 else np.ones(2)*1e-8
            spatial_ratio=eigen[0]/max(eigen[1],1e-8)
            temporal_support=(motion.loc[fid,'flow_inliers']>40 and np.linalg.norm(pose[1:]-prediction)<.6)
            valid=(q['error_px']<4.0 and q['inlier_fraction_6px']>.75 and q['n_boundary']>=100
                   and spread.min()>80 and (spatial_ratio>.015 or temporal_support) and 68<pose[0]<76)
            ranking=q['error_px']+.2*np.linalg.norm(pose[1:]-prediction)+max(0,.04-spatial_ratio)*20
            fits.append((not valid,ranking,pose,q,spatial_ratio))
        if fits:
            invalid,_,pose,q,spatial_ratio=min(fits,key=lambda z:(z[0],z[1]))
            valid=not invalid
        else:
            valid=False;pose=np.r_[72,prediction];q={'error_px':30,'inlier_fraction':0,'n_boundary':0};spatial_ratio=0
        if fid<293:
            # Initial position is constrained by the known start area even though the
            # nearly horizontal ground alone cannot fully identify camera x.
            valid=True
            pose=initial_pose.copy()
        if valid:
            offset=pose[1:]-raw;failures=0
        else:failures+=1
        rows.append(dict(frame=fid,time_s=motion.loc[fid,'time_s'],scale=pose[0],camera_x=pose[1],camera_y=pose[2],
                         valid=valid,error_px=q['error_px'],inlier_fraction=q['inlier_fraction'],
                         inlier_fraction_6px=q.get('inlier_fraction_6px',0),
                         n_boundary=q['n_boundary'],spatial_ratio=spatial_ratio))
        if len(rows)%10==0:print(fid,'pose',pose.round(2),'valid',valid,'error',round(q['error_px'],2),
                                'ratio',round(spatial_ratio,3),'failures',failures,flush=True)
    cap.release();pd.DataFrame(rows).to_csv(DEFAULT_OUT/'camera_keyframes.csv',index=False)
    print('accepted',sum(r['valid'] for r in rows),'of',len(rows),flush=True)


if __name__=='__main__':run()
