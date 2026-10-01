"""Track the rigid pot with NanoTrack and check it in both temporal directions.

No arms/hammer are included in the initialization rectangle. Scores are model
similarities, not calibrated probabilities. Model files are loaded locally.
"""
import hashlib
import json

import cv2
import numpy as np
import pandas as pd

from .extract_video_reference import DEFAULT_OUT, ROOT

MODELS=ROOT/'src/Data/video_reference/models/nanotrack_v2'


def make_tracker():
    params=cv2.TrackerNano_Params()
    params.backbone=str(MODELS/'nanotrack_backbone_sim.onnx')
    params.neckhead=str(MODELS/'nanotrack_head_sim.onnx')
    return cv2.TrackerNano_create(params)


def initial_box(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([20,160,40]),np.array([35,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c)
        if 45<w<100 and 60<h<110 and .55<w/h<1.3 and area/(w*h)>.5:
            candidates.append((area,(x,y,w,h)))
    if not candidates:raise RuntimeError('No starting pot found')
    return max(candidates)[1]


def run():
    cv2.setNumThreads(4)
    for name in ['nanotrack_backbone_sim.onnx','nanotrack_head_sim.onnx']:
        if not (MODELS/name).exists():raise FileNotFoundError(f'Local NanoTrack v2 model missing: {MODELS/name}')
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,203)
    ok,first=cap.read()
    if not ok:raise RuntimeError('Cannot decode the first tracking frame')
    tracker=make_tracker();bbox=initial_box(first);tracker.init(first,bbox)
    rows=[]
    for fid in range(203,1304):
        i=fid-203
        if i:
            ok,im=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {fid}')
        else:im=first
        if i:ok,bbox=tracker.update(im);score=float(tracker.getTrackingScore())
        else:ok=True;score=1.
        x,y,w,h=bbox;cx,cy=x+w/2,y+h/2
        size_ok=35<=w<=125 and 35<=h<=135 and x>=0 and y>=0 and x+w<=1280 and y+h<=720
        rows.append(dict(frame=fid,time_s=fid/fps,pixel_x=cx,pixel_y=cy,width=w,height=h,
                         score=score,valid=bool(ok and size_ok and score>=.8)))
        if fid%60==0:print('forward',fid,[cx,cy,w,h],round(score,3),flush=True)
    df=pd.DataFrame(rows).set_index('frame')
    cap.release()
    df.to_csv(DEFAULT_OUT/'player_rigid_forward.csv')
    # Reinitialize from a strong later observation and retrace it to the start.
    eligible=df[df.valid & (df.score>.82)]
    if eligible.empty:raise RuntimeError('No rigid-part anchor passed the appearance score gate')
    anchor=int(eligible.index[-1]);r=df.loc[anchor]
    tracker=make_tracker();box=(round(r.pixel_x-r.width/2),round(r.pixel_y-r.height/2),int(r.width),int(r.height))
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,anchor);ok,anchor_image=cap.read()
    if not ok:raise RuntimeError('Cannot decode backward anchor')
    tracker.init(anchor_image,box)
    df['backward_x']=np.nan;df['backward_y']=np.nan;df['backward_score']=np.nan
    df['backward_width']=np.nan;df['backward_height']=np.nan
    def reverse_frames():
        # Bound image memory to 60 frames instead of retaining the entire video.
        end=anchor
        while end>=203:
            begin=max(203,end-59);cap.set(cv2.CAP_PROP_POS_FRAMES,begin)
            chunk=[]
            for frame in range(begin,end+1):
                decoded,image=cap.read()
                if not decoded:raise RuntimeError(f'Cannot decode {frame}')
                chunk.append(image)
            for frame in range(end,begin-1,-1):yield frame,chunk[frame-begin]
            end=begin-1
    for fid,im in reverse_frames():
        if fid!=anchor:ok,box=tracker.update(im);score=float(tracker.getTrackingScore())
        else:ok=True;score=1.
        x,y,w,h=box
        if ok and 35<=w<=130 and 35<=h<=170:
            df.loc[fid,['backward_x','backward_y','backward_score','backward_width','backward_height']]=[x+w/2,y+h/2,score,w,h]
        if fid%90==0:print('backward',fid,[x+w/2,y+h/2,w,h],round(score,3),flush=True)
    cap.release()
    df['forward_backward_error_px']=np.hypot(df.pixel_x-df.backward_x,df.pixel_y-df.backward_y)
    df['bidirectional_valid']=df.valid&(df.backward_score>=.8)&(df.forward_backward_error_px<=18)
    df.to_csv(DEFAULT_OUT/'player_rigid.csv')
    provenance=dict(implementation='cv2.TrackerNano',opencv_version=cv2.__version__,
                    reference='https://docs.opencv.org/4.x/d8/d69/classcv_1_1TrackerNano.html',
                    models={name:dict(sha256=hashlib.sha256((MODELS/name).read_bytes()).hexdigest(),
                             source='https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2')
                            for name in ['nanotrack_backbone_sim.onnx','nanotrack_head_sim.onnx']},
                    init_box=initial_box(first),backward_anchor_frame=anchor,
                    forward_frames=int(df.valid.sum()),bidirectional_frames=int(df.bidirectional_valid.sum()))
    (DEFAULT_OUT/'rigid_tracker_report.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':run()
