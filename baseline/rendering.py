"""Native-layout RGB/depth/video exports with real camera conditioning."""
import copy,json,contextlib,math
from pathlib import Path
import numpy as np
import torch,imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
from gaussian_renderer import render
from baseline.checkpoint import load
from baseline.common import write
from baseline.evaluation import latest
from utils.exact_split_branchpoint import model_digest,digest

def byte(x):return (x.detach().clamp(0,1)*255+.5).byte().cpu().numpy().transpose(1,2,0)
def writer(p):return imageio.get_writer(str(p),fps=30,codec='libx264',macro_block_size=1,pixelformat='yuv420p',output_params=['-crf','18','-threads','2','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2'])
def camera_record(c):return dict(name=c.image_name,time=float(c.time),width=c.image_width,height=c.image_height,FoVx=c.FoVx,FoVy=c.FoVy,world_view_transform=c.world_view_transform.cpu().tolist(),camera_center=c.camera_center.cpu().tolist())

@torch.no_grad()
def generated_lookat(m,pattern,times):
    from scene.cameras import MiniCam
    from utils.lookat_utils import fibonacci_sphere
    from utils.graphics_utils import look_at,getProjectionMatrix
    x=m._xyz.detach().cpu().numpy();center=x.mean(0);bound=float(np.linalg.norm(x-center,axis=1).max());distance=1.25*bound/np.sin(min(pattern.FoVx,pattern.FoVy)/2)
    groups=[]
    for j,direction in enumerate(fibonacci_sphere(10)):
        up=np.array([0.,0.,-1.])
        if abs(np.dot(up,direction))>.99:up=np.array([0.,1.,0.])
        view=torch.as_tensor(look_at(center+direction*distance,center,up),dtype=torch.float32)
        near=.01;far=max(float(pattern.zfar),distance+3*bound);projection=getProjectionMatrix(near,far,pattern.FoVx,pattern.FoVy).T
        cam=MiniCam(pattern.image_width,pattern.image_height,pattern.FoVy,pattern.FoVx,near,far,view,view@projection,0.);cam.image_name=f'lookat_{j:02d}'
        sequence=[]
        for t in times:v=copy.copy(cam);v.time=t;sequence.append(v)
        groups.append((cam.image_name,sequence))
    return groups

@torch.no_grad()
def export(args):
    data=args.source_path.resolve();out=args.model_path.resolve();i=latest(out,args.iteration);m,s,p,b,c,state=load(out,data,i)
    before=(model_digest(m),digest(m.optimizer.state_dict()),digest(m.harmonic_nodes.payload() if hasattr(m,'harmonic_nodes') else None))
    groups=[]
    video=s.getVideoCameras() or sorted(s.getTrainCameras()+s.getTestCameras(),key=lambda cam:(float(cam.time),cam.image_name))
    if not args.skip_train:groups.append(('train',s.getTrainCameras()))
    if not args.skip_test:groups.append(('test',s.getTestCameras()))
    if not args.skip_video:groups.append(('video',video))
    if args.lookat:
        if state['is_c3vd']:lookats=generated_lookat(m,s.getTestCameras()[0],[float(c.time) for c in video])
        else:
            from baseline.data import NativeDataset
            folders=sorted((data/'lookat_cams').glob('lookat_*'));assert folders,'Dataset has no lookat camera folder'
            lookats=[(f.name,NativeDataset.readCamerasFromTransforms(None,str(f),'transforms.json',False)) for f in folders]
        groups += [('lookat_cameras/'+name,cams) for name,cams in lookats]
    counts={}
    for label,sequence in groups:
        cams=sequence[:args.max_frames] if args.max_frames else sequence
        dest=(out/'test'/f'ours_{i}'/label) if label.startswith('lookat_cameras/') else out/label/f'ours_{i}'
        dest.mkdir(parents=True,exist_ok=False)
        for folder in ['renders','depth','depth_raw']:(dest/folder).mkdir()
        if not label.startswith('lookat_cameras/'):
            for folder in ['gt','renders_no_dcol']:(dest/folder).mkdir()
        manifest=[]
        with contextlib.ExitStack() as stack:
            rgbstream=stack.enter_context(writer(dest/'ours_video_reconstruction.mp4'));depstream=stack.enter_context(writer(dest/'ours_depth.mp4'))
            for idx,cam in enumerate(tqdm(cams,desc=label)):
                q=render(cam,m,p,b,show_no_dcol=True);rgb=byte(q['render']);d=q['depth'].squeeze().float().cpu().numpy();dv=(np.clip(np.nan_to_num(d)/max(float(np.nanmax(d)),1e-6),0,1)*255).astype(np.uint8)
                Image.fromarray(rgb).save(dest/'renders'/f'{idx:05d}.png');Image.fromarray(dv).save(dest/'depth'/f'{idx:05d}.png');np.save(dest/'depth_raw'/f'{idx:05d}.npy',d)
                rgbstream.append_data(rgb);depstream.append_data(np.repeat(dv[:,:,None],3,axis=2))
                if not label.startswith('lookat_cameras/'):
                    Image.fromarray(byte(cam.original_image[:3])).save(dest/'gt'/f'{idx:05d}.png');Image.fromarray(byte(q['render_no_dcol'])).save(dest/'renders_no_dcol'/f'{idx:05d}.png')
                manifest.append(camera_record(cam))
        write(dest/'camera_manifest.json',manifest);counts[label]=len(cams)
    assert before==(model_digest(m),digest(m.optimizer.state_dict()),digest(m.harmonic_nodes.payload() if hasattr(m,'harmonic_nodes') else None))
    write(out/f'render_integrity_{i}.json',dict(pass_=True,read_only=True,counts=counts,partial=bool(args.max_frames),lookat_GT_available=not state['is_c3vd']))
