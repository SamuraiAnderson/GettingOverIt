"""Numerical checks for registration and honest treatment of missing observations."""
import unittest
import cv2
import numpy as np
import pandas as pd
from .extract_video_reference import project
from .video_edge_registration import EdgeMap
from .video_reference_export import camera_at_frames
from .video_path_completion import complete_short_gaps,TerrainPenetration
from .video_translation_track import repair_motion


class VideoReferenceTests(unittest.TestCase):
    def test_projection_roundtrip(self):
        world=np.array([[-44.,-3.],[60.,400.],[0.,0.]])
        s,x,y=72.,12.,220.
        uv=project(world,[s,x,y])
        recovered=np.column_stack([x+(uv[:,0]-640)/s,y-(uv[:,1]-360)/s])
        np.testing.assert_allclose(recovered,world,atol=1e-12)

    def test_synthetic_terrain_registration(self):
        p=np.array([[-4,-3],[-3,3],[0,2],[2,4],[5,-2]],dtype=np.float32)
        boundary=np.concatenate([a+(b-a)*np.linspace(0,1,120,endpoint=False)[:,None]
                                 for a,b in zip(p,np.roll(p,-1,axis=0))]).astype(np.float32)
        truth=np.array([72.,.4,.2]);im=np.zeros((720,1280,3),np.uint8)
        cv2.fillPoly(im,[np.round(project(p,truth)).astype(np.int32)],(200,200,200))
        atlas=EdgeMap([p],boundary,scale=72.,down=4)
        candidates=atlas.search(im,[1.2,-.4],radius=(2,2),top=3,prior_weight=0)
        fits=[atlas.refine(im,[72,c[1],c[2]]) for c in candidates]
        pose,q=min(fits,key=lambda pq:pq[1]['error_px'])
        self.assertLess(np.linalg.norm(pose[1:]-truth[1:]),.06)
        self.assertLess(q['error_px'],1.5)
        self.assertGreater(q['inlier_fraction'],.95)

    def test_gaps_are_not_invented(self):
        ids=np.arange(31)
        motion=pd.DataFrame(dict(camera_x=ids*.1,camera_y=ids*.2,flow_inliers=np.full(31,50)),index=ids)
        keys=pd.DataFrame([dict(frame=f,scale=72.,camera_x=f*.1+2,camera_y=f*.2+3,error_px=2.,valid=True)
                           for f in [0,9,30]])
        result=camera_at_frames(motion,keys,max_anchor_gap_frames=18)
        self.assertAlmostEqual(result.loc[5,'camera_x'],2.5)
        self.assertTrue(result.loc[10:29,'camera_x'].isna().all())
        motion.loc[4,'flow_inliers']=0
        result=camera_at_frames(motion,keys,max_anchor_gap_frames=18)
        self.assertTrue(result.loc[1:8,'camera_x'].isna().all())

    def test_short_gap_interpolation_has_provenance(self):
        times=np.arange(12)/30
        points=np.column_stack([times*3,np.ones(12)])
        points[:2]=np.nan;points[4:7]=np.nan;points[-2:]=np.nan
        filled,source,records=complete_short_gaps(times,points)
        np.testing.assert_allclose(filled[4:7,0],times[4:7]*3)
        self.assertTrue(np.all(source[4:7]=='interpolated'))
        self.assertTrue(np.isnan(filled[:2]).all())
        self.assertTrue(np.isnan(filled[-2:]).all())
        np.testing.assert_allclose(filled[3],points[3])

    def test_interpolation_rejects_wall_speed_and_long_gap(self):
        times=np.arange(5)/30
        points=np.array([[0.,0.],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[2.,0.]])
        wall=lambda p:((p[:,0]>.9)&(p[:,0]<1.1)).astype(float)
        filled,_,records=complete_short_gaps(times,points,wall)
        self.assertEqual(records[0]['reason'],'terrain_penetration')
        self.assertTrue(np.isnan(filled[1:4]).all())
        _,_,records=complete_short_gaps(times,points,max_speed=5)
        self.assertEqual(records[0]['reason'],'speed_jump')
        _,_,records=complete_short_gaps(times*10,points)
        self.assertEqual(records[0]['reason'],'long_gap')

    def test_camera_velocity_repair_keeps_raw_evidence(self):
        df=pd.DataFrame(dict(dx=[2.,np.nan,np.nan,5.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,5.],
                             dy=[1.,np.nan,np.nan,1.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,1.]))
        repaired=repair_motion(df)
        np.testing.assert_allclose(repaired.dx.iloc[1:3],[3.,4.])
        self.assertTrue(repaired.dx_raw.iloc[1:3].isna().all())
        self.assertTrue(repaired.flow_interpolated.iloc[1:3].all())
        self.assertTrue(repaired.dx.iloc[4:11].isna().all())

    def test_terrain_depth_uses_world_coordinates(self):
        rectangle=np.array([[2.,3.],[5.,3.],[5.,6.],[2.,6.]],np.float32)
        depth=TerrainPenetration([rectangle])
        values=depth(np.array([[3.5,4.5],[3.5,-4.5],[7.,4.5]]))
        self.assertGreater(values[0],1.4)
        np.testing.assert_array_equal(values[1:],[0.,0.])


if __name__=='__main__':unittest.main()
