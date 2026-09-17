"""Export observations and explicitly labelled, short-gap path interpolation."""
import hashlib
import json
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

from extract_video_reference import DEFAULT_OUT, ROOT, load_map, overlay_pose
from video_path_completion import TerrainPenetration,complete_short_gaps,interpolation_holdout


def camera_at_frames(motion, keyframes,max_anchor_gap_frames=90):
    """Integrate camera flow between map anchors, correcting accumulated drift.

    These are motion estimates between anchors, not direct map measurements.
    At most 3 s may be supported, and every flow loss must have been repaired by
    the explicit short velocity-gap policy. No camera extrapolation is done.
    """
    accepted=keyframes[keyframes.valid].sort_values('frame')
    result=pd.DataFrame(index=motion.index,columns=['scale','camera_x','camera_y','error_px'],dtype=float)
    for _,r in accepted.iterrows():
        result.loc[int(r.frame)]=r[['scale','camera_x','camera_y','error_px']].to_numpy(float)
    for (_,a),(_,b) in zip(accepted.iloc[:-1].iterrows(),accepted.iloc[1:].iterrows()):
        fa,fb=int(a.frame),int(b.frame)
        if fb-fa>max_anchor_gap_frames:continue
        ids=np.arange(fa,fb+1)
        # A lost camera motion interval is not evidence for a straight path.
        repaired=motion.loc[ids[1:],'flow_interpolated'] if 'flow_interpolated' in motion else pd.Series(False,index=ids[1:])
        if ((motion.loc[ids[1:],'flow_inliers']<16)&~repaired).any():continue
        weight=((ids-fa)/(fb-fa))[:,None]
        raw=motion.loc[ids,['camera_x','camera_y']].to_numpy(float)
        oa=a[['camera_x','camera_y']].to_numpy(float)-motion.loc[fa,['camera_x','camera_y']].to_numpy(float)
        ob=b[['camera_x','camera_y']].to_numpy(float)-motion.loc[fb,['camera_x','camera_y']].to_numpy(float)
        if np.linalg.norm(ob-oa)>max(1.5,(fb-fa)/30.):continue
        result.loc[ids,['camera_x','camera_y']]=raw+oa*(1-weight)+ob*weight
        result.loc[ids,'scale']=a.scale*(1-weight[:,0])+b.scale*weight[:,0]
        result.loc[ids,'error_px']=max(a.error_px,b.error_px)
    return result


def camera_holdout(motion,keys):
    errors=[]
    accepted=keys[keys.valid & (keys.time_s>10)].sort_values('frame')
    for _,anchor in accepted.iloc[1:-1].iterrows():
        hidden=keys.copy();hidden.loc[hidden.frame==anchor.frame,'valid']=False
        prediction=camera_at_frames(motion,hidden).loc[int(anchor.frame),['camera_x','camera_y']].to_numpy(float)
        if np.isfinite(prediction).all():
            errors.append(float(np.linalg.norm(prediction-anchor[['camera_x','camera_y']].to_numpy(float))))
    return dict(reference='held-out accepted map anchors, not game ground truth',predicted_anchors=len(errors),
                median_error_world=float(np.median(errors)) if errors else None,
                p90_error_world=float(np.percentile(errors,90)) if errors else None)


def run():
    out=DEFAULT_OUT
    motion=pd.read_csv(out/'camera_translation.csv').set_index('frame')
    keys=pd.read_csv(out/'camera_keyframes.csv')
    colors=pd.read_csv(out/'player_color.csv').set_index('frame')
    appearance=pd.read_csv(out/'player_v2.csv').set_index('frame')
    fused_path=out/'player_fused.csv'
    fused=pd.read_csv(fused_path).set_index('frame') if fused_path.exists() else None
    camera=camera_at_frames(motion,keys)
    delta=np.hypot(colors.pixel_x-appearance.pixel_x,colors.pixel_y-appearance.pixel_y)
    player_ok=(colors.confidence>=.65)&(colors.lost==0)&(appearance.lost==0)&(appearance.ncc>=.60)&(delta<=25)
    # Require detections on adjacent frames; one accidental agreement is insufficient.
    player_ok &= player_ok.shift(1,fill_value=False)|player_ok.shift(-1,fill_value=False)
    if fused is not None:player_ok=fused.valid
    camera_ok=camera.camera_x.notna()
    df=motion[['time_s']].copy()
    df['pixel_x']=(colors.pixel_x if fused is None else fused.pixel_x).where(player_ok)
    df['pixel_y']=(colors.pixel_y if fused is None else fused.pixel_y).where(player_ok)
    df['player_method']='colour_and_template' if fused is None else fused.source
    if fused is not None:
        df['rigid_tracking_score']=fused.score
        df['forward_backward_error_px']=fused.forward_backward_error_px
    df['player_candidate']=player_ok
    df['camera_candidate']=camera_ok
    df['camera_source']=np.where(camera_ok,'flow_between_map_anchors','missing')
    df.loc[df.index.isin(keys[keys.valid].frame),'camera_source']='map_anchor'
    df['camera_x']=camera.camera_x;df['camera_y']=camera.camera_y;df['scale_px_per_m']=camera.scale
    df['map_edge_error_px']=camera.error_px
    df['color_shape_score']=colors.confidence
    df['appearance_ncc']=appearance.ncc
    df['detector_disagreement_px']=delta
    ok=player_ok&camera_ok
    df['world_x']=(camera.camera_x+(df.pixel_x-640)/camera.scale).where(ok)
    df['world_y']=(camera.camera_y-(df.pixel_y-360)/camera.scale).where(ok)
    df['accepted_candidate']=ok
    df['observed_world_x']=df.world_x;df['observed_world_y']=df.world_y
    _,polygons,_=load_map();penetration=TerrainPenetration(polygons)
    observations=df[['world_x','world_y']].to_numpy(float)
    completed,sources,gaps=complete_short_gaps(df.time_s,observations,penetration)
    holdout=interpolation_holdout(df.time_s,observations,penetration)
    df[['world_x','world_y']]=completed
    df['point_source']=sources
    df['interpolated']=sources=='interpolated'
    df['has_position']=sources!='missing'
    (out/'interpolation_report.json').write_text(json.dumps(dict(holdout=holdout,gaps=gaps),indent=2,allow_nan=False))
    df['reason']=np.select([ok,~camera_ok&~player_ok,~camera_ok],
                           ['candidate_requires_review','camera_and_player_unresolved','camera_unresolved'],default='player_unresolved')
    df.loc[df.interpolated,'reason']='short_gap_interpolation'
    segment=-1;previous=-2;segments=[]
    df['segment_id']=-1
    for fid,r in df[df.has_position].iterrows():
        if fid!=previous+1:
            segment+=1;segments.append([])
        df.loc[fid,'segment_id']=segment
        segments[-1].append(dict(frame=int(fid),time_s=r.time_s,x=r.world_x,y=r.world_y,source=r.point_source))
        previous=fid
    df.to_csv(out/'trajectory.csv',index_label='frame')
    source=out/'source.mp4'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    info=dict(status='partial_unverified_draft',training_ready=False,
              source_video=str(source.relative_to(ROOT)),source_sha256=digest,
              map=str((ROOT/'checkpoints/environment.json').relative_to(ROOT)),
              map_sha256=hashlib.sha256((ROOT/'checkpoints/environment.json').read_bytes()).hexdigest(),
              coordinate_system='world x right, world y up; video reference is the visible pot centre',
              coordinate_warning='Pot visual centre is not calibrated to the game Player Transform origin.',
              actions_available=False,frame_range=[int(df.index.min()),int(df.index.max())],
              frames_evaluated=len(df),player_candidate_frames=int(player_ok.sum()),
              camera_candidate_frames=int(camera_ok.sum()),accepted_candidate_frames=int(ok.sum()),
              accepted_candidate_fraction=float(ok.mean()),map_keyframes=len(keys),
              interpolated_frames=int(df.interpolated.sum()),completed_frames=int(df.has_position.sum()),
              completed_fraction=float(df.has_position.mean()),interpolation_max_gap_s=.3,
              camera_max_anchor_gap_s=3.0,camera_flow_interpolation_max_missing_frames=6,
              camera_holdout=camera_holdout(motion,keys),
              interpolation_method='piecewise linear in world coordinates; speed/reversal/terrain gates; no extrapolation',
              interpolation_holdout=holdout,
              map_keyframes_accepted=int(keys.valid.sum()),
              median_accepted_map_error_px=float(camera.error_px[camera_ok].median()),
              ground_truth_accuracy='unknown; edge residual and agreement are quality signals, not ground truth',
              initial_seed='Known start area; initial camera fit from terrain, constant until frame 292.',
              video_frames_visually_reviewed_for_this_revision=False,
              segments=segments)
    (out/'reference_trajectory.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    make_plot(df,keys,out)
    make_overlay(df,out)
    summary={k:v for k,v in info.items() if k!='segments'}
    (out/'quality_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def make_plot(df,keys,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    _,polys,_=load_map()
    fig=plt.figure(figsize=(13,12),facecolor='#f5f7fa')
    grid=fig.add_gridspec(2,2,width_ratios=[.8,1.4],height_ratios=[1.5,1],wspace=.25,hspace=.25)
    ax=fig.add_subplot(grid[:,0]);detail=fig.add_subplot(grid[0,1]);quality=fig.add_subplot(grid[1,1])
    lines=[np.vstack([p,p[0]]) for p in polys]
    valid=df[df.has_position];observed=df[df.accepted_candidate]
    for plot in [ax,detail]:
        plot.add_collection(LineCollection(lines,colors='#8995a4',linewidths=.65))
        sc=plot.scatter(observed.world_x,observed.world_y,c=observed.time_s,cmap='plasma',s=9,zorder=3,vmin=df.time_s.min(),vmax=df.time_s.max())
        for _,segment in valid.groupby('segment_id'):
            if len(segment)>1:plot.plot(segment.world_x,segment.world_y,color='#cc3676',lw=1,zorder=2)
        filled=df[df.interpolated]
        plot.scatter(filled.world_x,filled.world_y,s=15,marker='x',color='#038b9b',zorder=4,label='Short-gap interpolation')
        plot.set_aspect('equal');plot.grid(alpha=.15);plot.set_xlabel('World x (m)');plot.set_ylabel('World y (m)')
    ax.set(xlim=(-58,100),ylim=(-12,450),title='Full terrain / accepted candidates only')
    if len(valid):
        detail.set(xlim=(valid.world_x.min()-8,valid.world_x.max()+8),ylim=(valid.world_y.min()-8,valid.world_y.max()+8))
    detail.set_title('Pot centre: colour = observations, cyan = interpolation')
    quality.plot(keys.time_s,keys.error_px,color='#748397',lw=1,label='Map residual')
    good=keys[keys.valid]
    quality.scatter(good.time_s,good.error_px,s=13,color='#178667',label='Map gate passed')
    quality.scatter(valid.time_s,np.full(len(valid),-.7),s=5,color='#c33576',label='Map + player candidate')
    quality.axhline(4,color='#b07222',ls='--',lw=1)
    quality.set(xlabel='Video time (s)',ylabel='Clipped mean edge distance (px)',ylim=(-1.5,16))
    quality.legend(loc='upper left',fontsize=8);quality.grid(alpha=.15)
    fig.colorbar(sc,ax=[ax,detail],fraction=.028,pad=.02,label='Video time (s)')
    fig.suptitle('Video-to-terrain reference: PARTIAL / NOT TRAINING-READY',fontsize=16,fontweight='bold')
    fig.text(.08,.035,f'{len(observed)} observed + {int(df.interpolated.sum())} interpolated / {len(df)} frames. Long gaps stay missing.\n'
             'No ground-truth position accuracy established; pot visual centre differs from Player origin.',fontsize=10,color='#506070')
    fig.savefig(out/'reference_trajectory.png',dpi=160,bbox_inches='tight');plt.close(fig)


def make_overlay(df,out):
    _,polys,_=load_map()
    cap=cv2.VideoCapture(str(out/'source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(df.index.min()))
    ffmpeg=shutil.which('ffmpeg')
    target=out/'reference_overlay.mp4'
    raw=out/'diagnostics/reference_overlay_mpeg4.mp4' if ffmpeg else target
    raw.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps/3,(1280,720))
    if not writer.isOpened():raise RuntimeError('Cannot create overlay video')
    frames_written=0
    for fid in df.index:
        ok,im=cap.read()
        if not ok:raise RuntimeError(f'Video ended at frame {fid}')
        if (fid-df.index.min())%3:continue
        r=df.loc[fid]
        if r.camera_candidate:im=overlay_pose(im,polys,[r.scale_px_per_m,r.camera_x,r.camera_y])
        if r.player_candidate:
            cv2.circle(im,(round(r.pixel_x),round(r.pixel_y)),18,(0,220,255),3)
        if r.interpolated and r.camera_candidate:
            pixel=(round((r.world_x-r.camera_x)*r.scale_px_per_m+640),round((r.camera_y-r.world_y)*r.scale_px_per_m+360))
            cv2.drawMarker(im,pixel,(240,80,240),cv2.MARKER_DIAMOND,22,2)
        label=f'{r.time_s:.2f}s  {r.reason}'
        cv2.rectangle(im,(0,0),(1280,74),(20,20,20),-1)
        cv2.putText(im,label,(18,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(im,'CV DRAFT | cyan: map | yellow: pot observation | magenta: interpolation | gaps kept',
                    (18,58),cv2.FONT_HERSHEY_SIMPLEX,.53,(130,200,255),1,cv2.LINE_AA)
        writer.write(im);frames_written+=1
    writer.release();cap.release()
    if ffmpeg:
        subprocess.run([ffmpeg,'-loglevel','error','-y','-i',str(raw),
                        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
                        str(target)],check=True)
    probe=cv2.VideoCapture(str(target))
    if int(probe.get(cv2.CAP_PROP_FRAME_COUNT))!=frames_written:raise AssertionError('Incomplete overlay export')
    probe.release()


if __name__=='__main__':run()
