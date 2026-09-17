"""Sparse whole-map candidates for re-localizing drifted camera motion."""
import cv2
import json
import numpy as np
from video_edge_registration import EdgeMap
from extract_video_reference import load_map, DEFAULT_OUT

def run():
    _,polygons,boundary=load_map()
    atlas=EdgeMap(polygons,boundary,down=8)
    cap=cv2.VideoCapture(str(DEFAULT_OUT/'source.mp4'));rows=[]
    fps=cap.get(cv2.CAP_PROP_FPS)
    for t in np.arange(12.8,41.,2.):
        fid=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode frame {fid}')
        cand=atlas.search(im,(40,215),(110,225),top=18,prior_weight=0)
        fits=[]
        for score,x,y,count in cand:
            p,q=atlas.refine(im,[atlas.scale,x,y])
            if q['n_boundary']<80:continue
            fits.append(dict(pose=p.tolist(),coarse=score,**q))
        fits.sort(key=lambda z:z['error_px'])
        print(f'{t:.1f}s: {len(fits)} global candidates',flush=True)
        rows.append(dict(frame=fid,time_s=fid/fps,candidates=fits))
    cap.release()
    (DEFAULT_OUT/'global_candidates.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':run()
