from pathlib import Path
import scene.endo_loader as endo
from scene.dataset_readers import sceneLoadTypeCallbacks,SceneInfo,getNerfppNorm,fetchPly
from utils.graphics_utils import BasicPointCloud
import numpy as np

NativeDataset=endo.ColonSplat_Dataset

class ProvidedSplitDataset(NativeDataset):
    def __init__(self,datadir,downsample=1.0,test_every=5):
        super().__init__(datadir,downsample=downsample,test_every=test_every)
        data=Path(datadir);train=list(self.all_cams)
        test=self.readCamerasFromTransforms(str(data/'test'),'transforms.json',False)
        for cam in train:cam.source_split='TRAIN';cam.source_dataset=str(data)
        for j,cam in enumerate(test):
            cam.original_image_name=cam.image_name;cam.uid=len(train)+j;cam.image_name='test__'+cam.image_name
            cam.source_split='TEST';cam.source_dataset=str(data/'test')
        self.all_cams=train+test;self.train_idxs=list(range(len(train)));self.test_idxs=list(range(len(train),len(train)+len(test)));self.video_idxs=list(range(len(self.all_cams)))

def read_c3vd(data,mode='monocular'):
    """Accept dataset input PLY, or regenerate initialization from its RGB/depth."""
    ds=endo.C3VD_Dataset(str(data),mode=mode);train=ds.format_infos('train');test=ds.format_infos('test')
    path=Path(data)/'points3d.ply'
    if path.exists():pcd=fetchPly(str(path));source='dataset points3d.ply (input data, not a model checkpoint)'
    else:
        xyz,rgb,normals=ds.get_init_pts();pcd=BasicPointCloud(points=xyz,colors=rgb,normals=normals);source='RGB/depth backprojection'
    return SceneInfo(point_cloud=pcd,train_cameras=train,test_cameras=test,video_cameras=None,
        nerf_normalization=getNerfppNorm(train),ply_path=str(path) if path.exists() else None,maxtime=ds.get_maxtime(),
        initialization_audit=dict(source=source,checkpoint_used=False,gt_ply_read=False,Gaussian_count=len(pcd.points)))

def configure(data):
    data=Path(data)
    if (data/'transforms.json').exists():
        assert (data/'test/transforms.json').exists(),'DynamicColon requires the supplied test folder'
        endo.ColonSplat_Dataset=ProvidedSplitDataset
        return False
    assert (data/'pose.txt').exists(),'Expected DynamicColon transforms.json or C3VD pose.txt'
    sceneLoadTypeCallbacks['c3vd']=read_c3vd
    return True
