import cv2
import numpy as np
import pandas as pd
from extract_video_reference import DEFAULT_OUT


def run(use_proposals=True):
    proposal_path=DEFAULT_OUT/'player_motion_proposals.csv'
    proposals=pd.read_csv(proposal_path).set_index('frame') if use_proposals and proposal_path.exists() else None
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,203)
    center=np.array([251.,234.]);velocity=np.zeros(2);hue=28.;area_ref=1100.;lost=0;rows=[];shape=None
    for fid in range(203,1304):
        ok,im=cap.read()
        if not ok:break
        small=cv2.resize(im,(640,360));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        prediction=center+velocity
        proposal=proposals.loc[fid,['proposal_x','proposal_y']].to_numpy(float)/2 if proposals is not None else np.full(2,np.nan)
        use_proposal=fid>300 and np.isfinite(proposal).all()
        if use_proposal and (lost>2 or np.linalg.norm(prediction-proposal)>120):
            prediction=proposal
        radius=75 if lost<4 else 250
        x0=max(0,int(prediction[0]-radius));x1=min(640,int(prediction[0]+radius))
        y0=max(0,int(prediction[1]-radius));y1=min(325,int(prediction[1]+radius))
        candidates=[]
        if x1>x0 and y1>y0:
            roi=hsv[y0:y1,x0:x1]
            hues=[hue-8,hue,hue+8,12,24,36,48,60,72,90,110,140] if lost>3 else [hue-8,hue,hue+8,24,36,48,60,72]
            for hh in np.unique(np.clip(hues,0,179)).astype(int):
                for saturation in [40,80,110,155]:
                    mask=cv2.inRange(roi,np.array([max(0,hh-9),saturation,15]),np.array([min(179,hh+9),255,255]))
                    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
                    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c)
                        if not (350<area<2300 and 20<w<60 and 23<h<65 and .55<w/h<1.55):continue
                        if area/(w*h)<.5:continue
                        m=cv2.moments(c);pos=np.array([m['m10']/m['m00']+x0,m['m01']/m['m00']+y0])
                        dist=np.linalg.norm(pos-prediction)
                        shaped=0 if shape is None else cv2.matchShapes(shape,c,cv2.CONTOURS_MATCH_I1,0)
                        confidence=np.exp(-.5*(np.log(area/area_ref)/.6)**2-2*shaped)
                        value=np.log(max(confidence,1e-8))-.5*(dist/(60 if lost>3 else 28))**2-(.0002 if lost>3 else .001)*(hh-hue)**2
                        if use_proposal:value-=.5*(np.linalg.norm(pos-proposal)/55)**2
                        candidates.append((value,confidence,pos,area,c,(x0,y0),hh))
        if candidates:
            value,conf,pos,area,c,origin,hh=max(candidates,key=lambda z:z[0])
            if conf>.45 and np.linalg.norm(pos-prediction)<95:
                velocity=.4*velocity+.6*(pos-center);center=pos;lost=0
                area_ref=.98*area_ref+.02*area
                smask=np.zeros(hsv.shape[:2],np.uint8)
                offset=c+np.array(origin,dtype=np.int32)
                cv2.fillPoly(smask,[offset],255)
                hv=hsv[:,:,0][smask>0]
                hue=.8*hue+.2*float(np.median(hv))
                if shape is None:shape=c.copy()
            else:lost+=1;velocity*=.5
        else:conf=0.;lost+=1;velocity*=.5
        rows.append(dict(frame=fid,time_s=fid/fps,pixel_x=2*center[0] if lost==0 else np.nan,
                         pixel_y=2*center[1] if lost==0 else np.nan,confidence=conf,hue=hue,lost=lost))
        if fid%30==0:print(fid,(2*center).round(1),'hue',round(hue,1),'confidence',round(conf,2),'lost',lost,flush=True)
    cap.release();pd.DataFrame(rows).to_csv(DEFAULT_OUT/'player_color.csv',index=False)


if __name__=='__main__':run()
