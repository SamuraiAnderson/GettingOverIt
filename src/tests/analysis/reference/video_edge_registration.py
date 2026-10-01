"""Raster chamfer registration of video edges to the static collider union."""
import cv2
import numpy as np
from scipy.optimize import minimize
from .extract_video_reference import load_map, frame_edges, project, DEFAULT_OUT


class EdgeMap:
    def __init__(self, polygons, boundary, scale=73.98570877, down=4):
        self.scale, self.down = scale, down
        self.res = scale / down
        self.boundary = boundary
        points = np.concatenate(polygons)
        self.lo = np.floor(points.min(0)) - 12
        self.hi = np.ceil(points.max(0)) + 12
        shape = np.ceil((self.hi-self.lo)*self.res).astype(int)[::-1]
        mask = np.zeros(shape, np.uint8)
        for p in polygons:
            q=np.column_stack([(p[:,0]-self.lo[0])*self.res,(self.hi[1]-p[:,1])*self.res])
            cv2.fillPoly(mask,[np.round(q).astype(np.int32)],255)
        self.edge=cv2.Canny(mask,50,100).astype(np.float32)/255
        self.h,self.w=720//down,1280//down
        self.channels=self.oriented(mask, self.edge)

    @staticmethod
    def oriented(gray, edge):
        gray=cv2.GaussianBlur(gray,(5,5),1)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0); gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
        theta=np.arctan2(gy,gx)%np.pi
        idx=np.floor(theta/np.pi*4+.5).astype(int)%4
        return np.stack([edge*(idx==i) for i in range(4)],axis=-1).astype(np.float32)

    def search(self, im, center, radius=(4,5), top=4, prior_weight=1.,min_edge_pixels=None):
        # Pixel coordinates of the world-space camera centre in the map raster.
        px=(center[0]-self.lo[0])*self.res;py=(self.hi[1]-center[1])*self.res
        rx,ry=np.asarray(radius)*self.res
        x0=max(0,int(px-self.w/2-rx)); y0=max(0,int(py-self.h/2-ry))
        x1=min(self.edge.shape[1],int(px+self.w/2+rx)+1)
        y1=min(self.edge.shape[0],int(py+self.h/2+ry)+1)
        if x1-x0<self.w or y1-y0<self.h:
            return []
        gray=cv2.cvtColor(cv2.resize(im,(self.w,self.h)),cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(3,3),.7)
        edge=cv2.Canny(gray,35,90)
        edge[int(650/self.down):,:int(335/self.down)]=0
        dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
        channels=self.oriented(gray,edge/255)
        od=[]
        for i in range(4):
            exact=cv2.distanceTransform((channels[:,:,i]==0).astype(np.uint8),cv2.DIST_L2,3)
            # Neighbour directions have a small cost instead of an absolute rejection.
            od.append(np.minimum(np.minimum(exact,dt+1.2),5))
        od=np.stack(od,axis=-1).astype(np.float32)
        valid=np.ones((self.h,self.w),np.float32)
        valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
        valid[int(650/self.down):,:int(335/self.down)]=0
        numerator=cv2.matchTemplate(self.channels[y0:y1,x0:x1],od*valid[:,:,None],cv2.TM_CCORR)
        count=cv2.matchTemplate(self.edge[y0:y1,x0:x1],valid,cv2.TM_CCORR)
        score=numerator/np.maximum(count,1)
        score[count<(800/self.down if min_edge_pixels is None else min_edge_pixels)]=100
        xxgrid=np.arange(score.shape[1])[None]+x0+self.w/2
        yygrid=np.arange(score.shape[0])[:,None]+y0+self.h/2
        score+=prior_weight*(.025*((xxgrid-px)/self.res)**2+.012*((yygrid-py)/self.res)**2)
        score+=30/self.down/np.maximum(count,1)
        candidates=[]
        for _ in range(top):
            flat=int(np.argmin(score)); yy,xx=np.unravel_index(flat,score.shape)
            if score[yy,xx]>=99:break
            cx=self.lo[0]+(x0+xx+self.w/2)/self.res
            cy=self.hi[1]-(y0+yy+self.h/2)/self.res
            candidates.append((float(score[yy,xx]),cx,cy,float(count[yy,xx])))
            cv2.circle(score,(xx,yy),int(self.res*.65),100,-1)
        return candidates

    def refine(self,im,pose):
        edge=frame_edges(im);dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
        uv=project(self.boundary,pose)
        take=(uv[:,0]>-120)&(uv[:,0]<1400)&(uv[:,1]>-120)&(uv[:,1]<840)
        points=self.boundary[take]
        def objective(p):
            xy=project(points,p)
            use=(xy[:,0]>12)&(xy[:,0]<1268)&(xy[:,1]>12)&(xy[:,1]<700)
            use &=~((xy[:,0]<335)&(xy[:,1]>650))
            xy=xy[use]
            if len(xy)<60:return 30.
            d=cv2.remap(dt,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
            return float(np.minimum(d,15).mean())
        s,x,y=pose
        best=(objective(pose),np.array(pose))
        for delta in [(0,0),(.08,0),(-.08,0),(0,.08),(0,-.08)]:
            opt=minimize(objective,[s,x+delta[0],y+delta[1]],method='Nelder-Mead',
                         bounds=[(s*.97,s*1.03),(x-.55,x+.55),(y-.55,y+.55)],
                         options={'maxiter':100,'xatol':.002,'fatol':.002})
            if .93*s<opt.x[0]<1.07*s and np.linalg.norm(opt.x[1:]-[x,y])<1 and opt.fun<best[0]:
                best=(float(opt.fun),opt.x)
        xy=project(points,best[1]);use=(xy[:,0]>12)&(xy[:,0]<1268)&(xy[:,1]>12)&(xy[:,1]<700)
        use &=~((xy[:,0]<335)&(xy[:,1]>650))
        xy=np.round(xy[use]).astype(int)
        distances=dt[xy[:,1],xy[:,0]]
        return best[1],dict(error_px=best[0],inlier_fraction=float(np.mean(distances<4)) if len(distances) else 0.,
                           inlier_fraction_6px=float(np.mean(distances<6)) if len(distances) else 0.,n_boundary=len(xy))
