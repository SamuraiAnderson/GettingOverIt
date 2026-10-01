"""Offline CV draft for the supplied Getting Over It speedrun video.

Run with --run in conda environment getting-over-it-analysis. No GUI, game,
training, or manual frame inspection is used. NanoTrack weights must be local.
The output is deliberately partial: unresolved positions remain missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT / 'src/Data/video_reference/37258923076-1-192'
SOURCE = Path('C:/Users/Symbol/Downloads/37258923076-1-192.mp4')


def load_map():
    env=json.loads((ROOT/'checkpoints/environment.json').read_text(encoding='utf-8-sig'))
    polygons=[np.asarray(p,np.float32) for c in env['colliders'] if c['name']!='Snake'
              for p in c['paths'] if len(p)>=3]
    points=np.concatenate(polygons);lo=np.floor(points.min(0))-2;hi=np.ceil(points.max(0))+2
    res=20.
    mask=np.zeros((int((hi[1]-lo[1])*res)+1,int((hi[0]-lo[0])*res)+1),np.uint8)
    for p in polygons:
        q=np.column_stack([(p[:,0]-lo[0])*res,(hi[1]-p[:,1])*res])
        cv2.fillPoly(mask,[np.round(q).astype(np.int32)],255)
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    pixels=np.concatenate([c[:,0][::2] for c in contours])
    boundary=np.column_stack([pixels[:,0]/res+lo[0],hi[1]-pixels[:,1]/res]).astype(np.float32)
    return env,polygons,boundary


def project(points,pose):
    scale,cx,cy=pose
    return np.column_stack([(points[:,0]-cx)*scale+640,(cy-points[:,1])*scale+360])


def frame_edges(im):
    gray=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(5,5),1.)
    edge=cv2.Canny(gray,70,150);edge[660:,:330]=0
    return edge


def overlay_pose(im,polygons,pose):
    out=im.copy()
    for poly in polygons:
        uv=project(poly,pose)
        if uv[:,0].max()>0 and uv[:,0].min()<1280 and uv[:,1].max()>0 and uv[:,1].min()<720:
            cv2.polylines(out,[np.round(uv).astype(np.int32)],True,(255,190,30),2,cv2.LINE_AA)
    return out


def prepare():
    DEFAULT_OUT.mkdir(parents=True,exist_ok=True)
    dest=DEFAULT_OUT/'source.mp4'
    if not dest.exists():shutil.copy2(SOURCE,dest)
    if SOURCE.exists() and hashlib.sha256(SOURCE.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
        raise RuntimeError('The project video differs from the supplied source; refusing to overwrite it.')
    cap=cv2.VideoCapture(str(dest))
    if not cap.isOpened():raise RuntimeError(f'Cannot decode {dest}')
    fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(3)),int(cap.get(4));cap.release()
    if (width,height)!=(1280,720) or not 29.9<fps<30.1 or frames<1304:
        raise ValueError('This configuration expects the supplied 1280x720, 30 fps video; recalibrate other videos.')
    (DEFAULT_OUT/'video_info.json').write_text(json.dumps(dict(fps=fps,frames=frames,duration_s=frames/fps,
                                                             width=width,height=height),indent=2))


def calibrate():
    _,_,boundary=load_map()
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,203)
    ok,im=cap.read();cap.release()
    if not ok:raise RuntimeError('Cannot decode the initial gameplay frame')
    dt=cv2.distanceTransform(255-frame_edges(im),cv2.DIST_L2,3)
    def objective(p):
        uv=project(boundary,p)
        use=(uv[:,0]>12)&(uv[:,0]<1268)&(uv[:,1]>12)&(uv[:,1]<680)
        use &=~((uv[:,0]<330)&(uv[:,1]>650))
        xy=uv[use]
        if len(xy)<100:return 30.
        d=cv2.remap(dt,xy[:,0][None].astype(np.float32),xy[:,1][None].astype(np.float32),cv2.INTER_LINEAR)[0]
        return float(np.minimum(d,15).mean())
    opt=differential_evolution(objective,[(35,110),(-49.5,-35.5),(-6.5,3.5)],
                               seed=5,popsize=18,maxiter=120,tol=.0005,polish=True)
    result=dict(pose=opt.x.tolist(),score=float(opt.fun),seed='known start area',frame=203)
    (DEFAULT_OUT/'initial_pose.json').write_text(json.dumps(result,indent=2))
    print('Initial terrain alignment:',result,flush=True)


def run_pipeline():
    prepare()
    if not (DEFAULT_OUT/'initial_pose.json').exists():calibrate()
    from .video_translation_track import run as motion
    from .video_global_registration import run as global_match
    from .video_camera_register import run as register
    from .video_player_color import run as color
    from .video_follow_camera import run as motion_proposal
    from .video_player_track import run as appearance
    from .video_player_rigid import run as rigid
    from .video_player_fusion import run as fuse
    from .video_reference_export import run as export
    for name,operation in [('camera motion',motion),('global edge candidates',global_match),('camera registration',register),
                           ('player colour bootstrap',lambda:color(use_proposals=False)),('camera follower proposal',motion_proposal),
                           ('player colour refinement',lambda:color(use_proposals=True)),('player appearance cross-check',appearance),
                           ('rigid pot tracking',rigid),('pot centre fusion',fuse),
                           ('export',export)]:
        print(f'Running {name}',flush=True);operation()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true',help='Recompute the complete offline CV draft')
    group.add_argument('--export',action='store_true',help='Export existing intermediate detections')
    group.add_argument('--calibrate',action='store_true',help='Fit the known starting scene')
    args=parser.parse_args()
    if args.run:run_pipeline()
    elif args.calibrate:prepare();calibrate()
    else:
        from .video_reference_export import run
        run()
