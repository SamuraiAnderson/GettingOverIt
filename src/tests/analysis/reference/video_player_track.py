"""Gold-pot initialization and masked appearance tracking, entirely offline."""
import cv2
import numpy as np
import pandas as pd
from .extract_video_reference import DEFAULT_OUT


def appearance(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.,(8,8)).apply(gray)


def run():
    colors=pd.read_csv(DEFAULT_OUT/'player_color.csv').set_index('frame')
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,203)
    rows=[];center=None;velocity=np.zeros(2);original=None;recent=None;size=None;lost=0
    for fid in range(203,1304):
        ok,im=cap.read()
        if not ok:break
        gray=appearance(im)
        if center is None:
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,np.array([20,160,40]),np.array([35,255,255]))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            boxes=[]
            for c in contours:
                x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c)
                if 45<w<100 and 60<h<110 and .55<w/h<1.3 and area/(w*h)>.5:
                    boxes.append((area,x,y,w,h))
            if not boxes:raise RuntimeError('No unique pot initialization found')
            _,x,y,w,h=max(boxes)
            center=np.array([x+w/2,y+h/2]);size=np.array([w,h])
            # Keep the interior, reducing terrain pixels near the silhouette.
            original=gray[y+8:y+h-7,x+8:x+w-8].copy();recent=original.copy()
        else:
            prediction=center+velocity
            color=colors.loc[fid]
            color_pos=color[['pixel_x','pixel_y']].to_numpy(float)
            color_available=np.isfinite(color_pos).all() and color.confidence>.65
            if color_available and (np.linalg.norm(color_pos-center)<180 or lost>3):
                prediction=color_pos
            margin=110 if lost<3 else 230
            x0=max(0,int(prediction[0]-margin));x1=min(1280,int(prediction[0]+margin))
            y0=max(0,int(prediction[1]-margin));y1=min(720,int(prediction[1]+margin))
            roi=gray[y0:y1,x0:x1]
            best=None
            templates=[(recent,0),(original,1)]
            for template,source in templates:
                for angle in [-135,-90,-45,0,45,90,135,180]:
                    hh,ww=template.shape
                    mat=cv2.getRotationMatrix2D(((ww-1)/2,(hh-1)/2),angle,1)
                    templ=cv2.warpAffine(template,mat,(ww,hh),borderMode=cv2.BORDER_REFLECT_101)
                    if roi.shape[0]<hh or roi.shape[1]<ww:continue
                    response=cv2.matchTemplate(roi,templ,cv2.TM_CCOEFF_NORMED)
                    yy,xx=np.indices(response.shape)
                    distance=((xx+x0+ww/2-prediction[0])**2+(yy+y0+hh/2-prediction[1])**2)**.5
                    weighted=response-.10*np.minimum(distance/80,2)
                    iy,ix=np.unravel_index(np.argmax(weighted),weighted.shape)
                    conf=float(response[iy,ix]);value=float(weighted[iy,ix])-.025*source
                    if best is None or value>best[0]:
                        best=(value,conf,np.array([x0+ix+ww/2,y0+iy+hh/2]),source)
            if best is not None:
                _,conf,pos,source=best
            else:conf=0.;pos=center.copy();source=-1
            agree=color_available and np.linalg.norm(pos-color_pos)<35
            if conf>.60 or (conf>.40 and agree):
                velocity=.4*velocity+.6*(pos-center);center=pos;lost=0
                # Update only a strong match that also resembles the first template.
                hh,ww=recent.shape;x=int(center[0]-ww/2);y=int(center[1]-hh/2)
                if agree and fid%3==0 and x>=0 and y>=0 and x+ww<=1280 and y+hh<=720:
                    recent=gray[y:y+hh,x:x+ww].copy()
            else:
                lost+=1;velocity*=.5
            if fid%30==0:
                print(fid,center.round(1),'ncc',round(conf,3),'lost',lost,flush=True)
        if fid==203:conf=1.;source=1;agree=True
        rows.append(dict(frame=fid,time_s=fid/fps,pixel_x=center[0] if lost==0 else np.nan,
                         pixel_y=center[1] if lost==0 else np.nan,ncc=conf,lost=lost,template_source=source,
                         color_agreement=agree))
    cap.release();pd.DataFrame(rows).to_csv(DEFAULT_OUT/'player_v2.csv',index=False)


if __name__=='__main__':run()
