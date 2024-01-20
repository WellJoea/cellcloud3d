import os
from os.path import join as pjoin
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align import affine_registration, register_dwi_to_template
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   RigidIsoScalingTransform3D,
                                   RigidScalingTransform3D,
                                   AffineTransform3D)

from ..plotting import qview, comp_images

def dipyreg():
    def __init__(self, transtype=None):
        self.transtype = ['rigid'] if transtype is None else transtype

    def regist_affine(self, static, moving,
                      moving_affine=None,
                      static_affine=None,
                      nbins=32,
                      metric='MI',
                      pipeline=None,
                      level_iters = [10000, 1000, 100],
                      sigmas = [3.0, 1.0, 0.0],
                      factors = [4, 2, 1],
                      **kargs):
        xformed_img, reg_affine = affine_registration(
                moving,
                static,
                moving_affine=moving_affine,
                static_affine=static_affine,
                nbins=32,
                metric='MI',
                pipeline=pipeline,
                level_iters=level_iters,
                sigmas=sigmas,
                factors=factors, **kargs)

def test():
    rootdir = '/share/home/zhonw/WorkSpace/11Project/04GNNST/'
    workdir = f'{rootdir}/02Analysis/03LM_Visum_E10.5/20230625_E10.5_total'
    datadir = f'{rootdir}/01DataBase/03LM_Visum_E10.5/20230625_E10.5_total'

    os.chdir(workdir)
    os.makedirs(f'{workdir}/regist', exist_ok=True)
    os.chdir(workdir)
    print(workdir)

    curdir = f"{workdir}/regist"
    #imageR = ski.io.imread(f'{curdir}/all_image_2000.2000.tif')
    imageL = iio.imread(f'{curdir}/all_image_2000.2000.stack.rgb.tif') # index=(...)
    tmats = np.load(f'{curdir}/all_image_2000.2000.stack.tmats.npy')
    imageb = ski.color.rgb2gray(imageL)

    imageM = iio.imread(f'/share/home/zhonw/WorkSpace/11Project/04GNNST/02Analysis/02LM_Visium/stak.reg.rgb.2000.2000.tif') # index=(...)
    imageMb = ski.color.rgb2gray(imageM)
    imageMb = np.array([cc.tf.mirrorhv(img, x=True, y=True) for img in imageMb])
    imageMb = imageMb[::-1,...]

    # Load Elastix Image Filter Object
    def resize3d(imged, rsize, interpolation=cv2.INTER_AREA):
        imged = imged.copy()
        imgn = [ cv2.resize(i, rsize, interpolation=interpolation) for i in imged]
        return np.array(imgn).astype(imged.dtype)

    fixed_image = resize3d(imageb, (500,500))
    moving_image= resize3d(imageMb, (500,500))

    zscale1 = (1240/2000) *10 *(500/5380)
    zscale2 = (1240/2000) *100 *(500/4785)
    zscale1, zscale2

    import matplotlib
    %matplotlib inline
    from ipywidgets import interact,fixed
    interact(
        comp_images,
        fixed_image_z=(0, fixed_image.shape[0] - 1),
        moving_image_z=(0, moving_image.shape[0] - 1),
        axis=fixed(0),
        fixed_npa=fixed(fixed_image),
        moving_npa=fixed(moving_image),

    )

    # fixed_image = sitk.GetImageFromArray(fixed_image.astype(np.float32), isVector = None)
    # moving_image = sitk.GetImageFromArray(moving_image.astype(np.float32), isVector = None)
    # fixed_image.SetOrigin((0,0,0))
    # fixed_image.SetSpacing([1, 1, zscale1])

    # moving_image.SetOrigin((0,0,0))
    # moving_image.SetSpacing([1, 1, zscale2])

    # cc.ut.Info(fixed_image)
    # cc.ut.Info(moving_image)
    # #pltshow(*sitk.GetArrayViewFromImage(fixed_image), *sitk.GetArrayViewFromImage(moving_image))

    # sitk.WriteImage(fixed_image, 'aa.nii')
    # sitk.WriteImage(moving_image, 'bb.nii')

    # import nibabel as nib
    # static_data, static_affine, static_img = load_nifti(f'{workdir}/aa.nii', return_img=True)
    # static = static_data
    # static_grid2world = static_affine

    # moving_data, moving_affine, moving_img = load_nifti(f'{workdir}/bb.nii',return_img=True)
    # moving = moving_data
    # moving_grid2world = moving_affine


    from dipy.align.imaffine import (transform_centers_of_mass,
                                    AffineMap,
                                    MutualInformationMetric,
                                    AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                    RigidTransform3D,
                                    RigidIsoScalingTransform3D,
                                    RigidScalingTransform3D,
                                    AffineTransform3D)

    static = np.transpose(fixed_image, axes=(1,2,0))
    static_grid2world = np.eye(4)
    static_grid2world[2,2] = zscale1

    moving = np.transpose(moving_image, axes=(1,2,0))
    moving_grid2world = np.eye(4)
    moving_grid2world[2,2] = zscale2

    static.shape, static_grid2world, moving.shape, moving_grid2world


    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                        moving, moving_grid2world)

    transformed = c_of_mass.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_com_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_com_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_com_2.png")



    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)


    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)

    transformed = translation.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_trans_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_trans_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_trans_2.png")



    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigid.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_rigid_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_rigid_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_rigid_2.png")


    from dipy.align.transforms import (TranslationTransform3D,
                                    RigidTransform3D,
                                    RigidIsoScalingTransform3D,
                                    RigidScalingTransform3D,
                                    ScalingTransform3D,
                                    AffineTransform3D)
    transform = RigidIsoScalingTransform3D()
    params0 = None
    starting_affine = rigid.affine
    #starting_affine = translation.affine
    rigidiso = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigidiso.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_affine_2.png")

    #RigidIsoScalingTransform3D/RigidScalingTransform3D
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigidiso.affine
    affine = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = affine.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_affine_2.png")


    import numpy as np
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    from dipy.data import get_fnames
    from dipy.io.image import load_nifti
    from dipy.viz import regtools

    metric = CCMetric(3)

    level_iters = [15, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)


    static_affine = static_grid2world
    moving_affine = moving_grid2world
    pre_align = affine.affine

    mapping = sdr.optimize(static,
                        moving, 
                        static_affine, 
                        moving_affine, 
                        pre_align)
    warped_moving = mapping.transform(moving)
    regtools.overlay_slices(static, warped_moving, None, 2, 'Static',
                            'Warped moving', 'warped_moving.png')
    warped_static = mapping.transform_inverse(static)
    regtools.overlay_slices(warped_static, moving, None, 1, 'Warped static',
                            'Moving', 'warped_static.png')

    # %matplotlib inline
    # from ipywidgets import interact,fixed
    # interact(
    #     cc.pl.comp_images,
    #     fixed_image_z=(0, warped_moving.shape[2] - 1),
    #     moving_image_z=(0, warped_static.shape[2] - 1),
    #     axis=fixed(2),
    #     fixed_npa=fixed(warped_moving),
    #     moving_npa=fixed(warped_static),

    # )
    # %matplotlib inline
    # from ipywidgets import interact,fixed
    # interact(
    #     cc.pl.comp_images,
    #     fixed_image_z=(0, moving.shape[2] - 1),
    #     moving_image_z=(0, warped_static.shape[2] - 1),
    #     axis=fixed(2),
    #     fixed_npa=fixed(moving),
    #     moving_npa=fixed(warped_static),

    # )


    from dipy.align import affine_registration, register_dwi_to_template
    from dipy.align import (affine_registration, translation, rigid_isoscaling,
                            rigid_scaling,affine, center_of_mass,
                            rigid, register_series)

    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    # ["center_of_mass",
    # "translation",
    # "rigid_isoscaling",
    # "rigid_scaling",
    # "rigid",
    # "affine"]

    pipeline = []
    sampling_prop = None
    static_affine = None
    moving_affine = None

    nbins = 32
    sampling_proportion = None
    metric = 'MI'

    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    moving = matched
    static = static
    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=moving_affine,
        static_affine=static_affine,
        nbins=nbins,
        metric=metric,
        pipeline=pipeline,
        level_iters=level_iters,
        sampling_proportion = sampling_proportion,
        sigmas=sigmas,    
        factors=factors)

    cc.pl.qview(xformed_img, moving, static)

    from dipy.align import affine_registration, register_dwi_to_template

    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    # ["center_of_mass",
    # "translation",
    # "rigid_isoscaling",
    # "rigid_scaling",
    # "rigid",
    # "affine"]

    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=moving_affine,
        static_affine=static_affine,
        nbins=32,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)

    regtools.overlay_slices(static, xformed_img, None, 0,
                            "Static", "Transformed", "xformed_affine_0.png")
    regtools.overlay_slices(static, xformed_img, None, 1,
                            "Static", "Transformed", "xformed_affine_1.png")
    regtools.overlay_slices(static, xformed_img, None, 2,
                            "Static", "Transformed", "xformed_affine_2.png")


    # xformed_dwi, reg_affine = register_dwi_to_template(
    #     dwi=static_img,
    #     gtab=(pjoin(folder, 'HARDI150.bval'),
    #           pjoin(folder, 'HARDI150.bvec')),
    #     template=moving_img,
    #     reg_method="aff",
    #     nbins=32,
    #     metric='MI',
    #     pipeline=pipeline,
    #     level_iters=level_iters,
    #     sigmas=sigmas,
    #     factors=factors)

    # regtools.overlay_slices(moving, xformed_dwi, None, 0,
    #                         "Static", "Transformed", "xformed_dwi_0.png")
    # regtools.overlay_slices(moving, xformed_dwi, None, 1,
    #                         "Static", "Transformed", "xformed_dwi_1.png")
    # regtools.overlay_slices(moving, xformed_dwi, None, 2,
    #                         "Static", "Transformed", "xformed_dwi_2.png")