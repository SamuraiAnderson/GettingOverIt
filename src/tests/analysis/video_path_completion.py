"""Short-gap interpolation in world space with provenance and terrain checks."""
import cv2
import numpy as np


class TerrainPenetration:
    """Distance inside the collider union, sampled at 0.05 world-unit resolution."""
    def __init__(self,polygons,resolution=20):
        self.res=resolution
        all_points=np.concatenate(polygons)
        self.lo=np.floor(all_points.min(0))-2;self.hi=np.ceil(all_points.max(0))+2
        size=np.ceil((self.hi-self.lo)*resolution).astype(int)
        solid=np.zeros((size[1]+1,size[0]+1),np.uint8)
        for p in polygons:
            q=np.column_stack([(p[:,0]-self.lo[0])*resolution,(self.hi[1]-p[:,1])*resolution])
            cv2.fillPoly(solid,[np.round(q).astype(np.int32)],255)
        self.depth=cv2.distanceTransform(solid,cv2.DIST_L2,3)/resolution

    def __call__(self,points):
        q=np.column_stack([(points[:,0]-self.lo[0])*self.res,(self.hi[1]-points[:,1])*self.res])
        return cv2.remap(self.depth,q[:,0][None].astype(np.float32),q[:,1][None].astype(np.float32),
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]


def complete_short_gaps(times,observed,penetration=None,max_gap_s=.3,max_speed=80.,max_penetration=.20):
    """Linear interpolation never extrapolates, smooths observations, or bridges long gaps.

    Linear segments avoid cubic overshoot near impact events. A dense spatial
    test rejects terrain shortcuts; speed checks are heuristic quality gates,
    not a reconstruction of the underlying hammer physics.
    """
    times=np.asarray(times,float);observed=np.asarray(observed,float)
    result=observed.copy();valid=np.isfinite(observed).all(1)
    source=np.where(valid,'observed','missing').astype(object)
    records=[];ids=np.flatnonzero(valid)
    for a,b in zip(ids[:-1],ids[1:]):
        if b==a+1:continue
        dt=times[b]-times[a];distance=np.linalg.norm(observed[b]-observed[a])
        record=dict(start_index=int(a),end_index=int(b),start_s=float(times[a]),end_s=float(times[b]),
                    missing_frames=int(b-a-1),duration_s=float(dt),status='rejected')
        if dt<=0 or dt>max_gap_s+1e-8:record['reason']='long_gap';records.append(record);continue
        if distance/dt>max_speed:record['reason']='speed_jump';records.append(record);continue
        slope=(observed[b]-observed[a])/dt
        # Reject a sharp unexplained reversal across the missing interval.
        neighbours=[]
        if a>0 and valid[a-1]:neighbours.append((observed[a]-observed[a-1])/(times[a]-times[a-1]))
        if b+1<len(times) and valid[b+1]:neighbours.append((observed[b+1]-observed[b])/(times[b+1]-times[b]))
        if any(np.linalg.norm(v-slope)>max_speed*.65 for v in neighbours):
            record['reason']='velocity_discontinuity';records.append(record);continue
        dense=np.linspace(observed[a],observed[b],max(3,int(distance/.04)+1))
        if penetration is not None and float(np.max(penetration(dense)))>max_penetration:
            record['reason']='terrain_penetration';records.append(record);continue
        w=((times[a+1:b]-times[a])/dt)[:,None]
        result[a+1:b]=observed[a]*(1-w)+observed[b]*w
        source[a+1:b]='interpolated'
        record.update(status='filled',reason='short_gap_with_supported_endpoints',speed=float(distance/dt))
        records.append(record)
    return result,source,records


def interpolation_holdout(times,points,penetration,max_gap_s=.3):
    """Compare hidden observations with interpolation; CV observations are not ground truth."""
    times=np.asarray(times);points=np.asarray(points);valid=np.isfinite(points).all(1)
    report=[]
    for length in [2,4,6]:
        errors=[];attempts=0;filled=0
        for begin in range(5,len(points)-length-5,19):
            if not valid[begin-2:begin+length+2].all():continue
            attempts+=1;hidden=points.copy();hidden[begin:begin+length]=np.nan
            restored,source,_=complete_short_gaps(times,hidden,penetration,max_gap_s=max_gap_s)
            keep=np.isfinite(restored[begin:begin+length]).all(1)
            if not keep.all():continue
            filled+=1;errors.extend(np.linalg.norm(restored[begin:begin+length]-points[begin:begin+length],axis=1).tolist())
        report.append(dict(hidden_frames=length,attempted_windows=attempts,filled_windows=filled,
                           median_error_world=float(np.median(errors)) if errors else None,
                           p90_error_world=float(np.percentile(errors,90)) if errors else None))
    return dict(reference='held-out CV observations, not game ground truth',max_gap_s=max_gap_s,results=report)
