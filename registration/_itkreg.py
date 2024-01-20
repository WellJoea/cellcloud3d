import itk 
import numpy as np
import skimage as ski
from skimage import transform as skitf

class itkregist():
    def __init__(self, transtype=None, resolutions=None, GridSpacing=None, verb=False):
        self.transtype = ['rigid'] if transtype is None else transtype
        self.params = self.initparameters(trans=self.transtype,
                                                         resolutions=resolutions, 
                                                         GridSpacing=GridSpacing,
                                                         verb=verb)

        if verb:
            print(self.params)

    def itkinit(self, fixed_img,
                 moving_img,
                 parameter_object,
                fixed_mask=None,
                moving_mask=None,
                log_to_console=False,
                output_directory=None,
                number_of_threads=10, **kargs):
        elastix_object = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
        if not fixed_mask is None:
            elastix_object.SetFixedMask(fixed_mask)
        if not moving_mask is None:
            elastix_object.SetMovingMask(moving_mask)

        elastix_object.SetParameterObject(parameter_object)
        # elastix_object.SetInitialTransformParameterFileName('')
        elastix_object.SetNumberOfThreads(number_of_threads)
        elastix_object.SetLogToConsole(log_to_console)

        if not output_directory is None:
            elastix_object.SetOutputDirectory(output_directory)
        #elastix_object.SetOutputDirectory('./exampleoutput/')
        # elastix_object.SetComputeSpatialJacobian(True)
        # elastix_object.SetComputeDeterminantOfSpatialJacobian(True)

        elastix_object.UpdateLargestPossibleRegion()
        # elastix_object.Update()

        result_image = elastix_object.GetOutput()
        result_transform_parameters = elastix_object.GetTransformParameterObject()

        return result_image, result_transform_parameters

    def itkmehtod(self, fixed_gray,
                    moving_gray,
                    parameter_object,
                    log_to_file=False,
                    log_file_name="regist.log",
                    number_of_threads=10,
                    output_directory='.', 
                    log_to_console= False,
                    **kargs):
        mov_out, paramsnew = itk.elastix_registration_method(
                                fixed_gray,
                                moving_gray,
                                parameter_object=parameter_object,
                                log_to_file=log_to_file,
                                log_file_name=log_file_name,
                                number_of_threads=number_of_threads,
                                output_directory=output_directory, 
                                log_to_console= log_to_console,
                                **kargs)
        return mov_out, paramsnew

    def regist(self, 
                fixed_itk,
                mov_itk,
                params=None, 
                log_to_file=True,
                isscale=None,
                log_file_name="regist.log",
                number_of_threads=5,
                output_directory=None,
                log_to_console=False,
                **kargs,
    ):
        if not isinstance(fixed_itk, itk.Image):
            # isscale = self.onechannel(fixed_itk) if isscale is None else isscale
            # fixed_gray = fixed_img if self.onechannel(fixed_img) else ski.color.rgb2gray(fixed_img)
            # moving_gray = moving_img if self.onechannel(moving_img) else ski.color.rgb2gray(moving_img)

            fixed_itk = itk.GetImageFromArray(np.ascontiguousarray(fixed_itk.astype(np.float32)))
            mov_itk = itk.GetImageFromArray(np.ascontiguousarray(mov_itk.astype(np.float32)))

        params = self.params if params is None else params
        mov_out, paramsnew = self.itkinit(fixed_itk,
                                            mov_itk,
                                            params,
                                            log_to_file=log_to_file,
                                            log_file_name=log_file_name,
                                            number_of_threads=number_of_threads,
                                            output_directory=output_directory, 
                                            log_to_console= log_to_console, **kargs)
        ntrans = paramsnew.GetNumberOfParameterMaps()
        self.tmat = [ self.get_transmtx(paramsnew, map_num=i ) for i in range(ntrans)]
        self.mov_out = mov_out
        self.paramsnew = paramsnew
        self.moving_img = mov_itk
        return self

    def transform(self,  moving_img=None, paramsnew=None):
        moving_img = self.moving_img if moving_img is None else moving_img
        paramsnew = self.paramsnew if paramsnew is None else paramsnew
        return itk.transformix_filter( moving_img, paramsnew)

    @staticmethod
    def get_transmtx(params, map_num=0 ):
        #https://elastix.lumc.nl/doxygen/classelastix_1_1SimilarityTransformElastix.html
        fixdim = params.GetParameter(map_num, 'FixedImageDimension')[0]
        tranty = params.GetParameter(map_num, 'Transform')[0]
        center = np.asarray(params.GetParameter(map_num, 'CenterOfRotationPoint')).astype(np.float64)
        trans = np.asarray(params.GetParameter(map_num, 'TransformParameters')).astype(np.float64)

        tform = None
        if (fixdim=='2') and (tranty=='EulerTransform'):
            tform = skitf.EuclideanTransform(rotation=trans[0],
                                             translation=trans[1:3]).params.astype(np.float64)
            shif = skitf.EuclideanTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='2') and (tranty=='SimilarityTransform'):
            tform = skitf.SimilarityTransform(scale=trans[0],
                                              rotation=trans[1],
                                              translation=trans[2:4]).params.astype(np.float64)
            shif = skitf.SimilarityTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='2') and (tranty=='AffineTransform'):
            tform = np.eye(3).astype(np.float64)
            tform[:2, :2] = np.array(trans[:4]).reshape(2,2)
            tform[:2, 2]  = trans[4:6]
            shif = skitf.AffineTransform(translation=center, dimensionality=2).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        elif (fixdim=='3') and (tranty=='AffineTransform'):
            tform = np.eye(4).astype(np.float64)
            tform[:3, :3] = np.array(trans[:9]).reshape(3,3)
            tform[:3, 3]  = trans[9:12]
            shif = skitf.SimilarityTransform(translation=center, dimensionality=3).params.astype(np.float64)
            tform = shif @ tform @ np.linalg.inv(shif)
        else:
            tform = np.eye(3).astype(np.float64)
        return tform.astype(np.float64)

    @staticmethod
    def str2list(lists, values=None):
        if lists is None:
            lists = values
        if type(lists) in [str, int, float, bool]:
            return [lists]
        else:
            return lists

    @staticmethod
    def onechannel(image):
        if image.ndim==2:
            return True
        elif image.ndim==3:
            return False
        else:
            raise ValueError('the image must have 2 or 3 dims.')

    @staticmethod
    def scaledimg(images):
        if (np.issubdtype(images.dtype, np.integer) or
            (images.dtype in [np.uint8, np.uint16, np.uint32, np.uint64])) and \
            (images.max() > 1):
            return False
        else:
            return True

    @staticmethod
    def initparameters(trans=None, resolutions=None, GridSpacing=None, verb=False):
        parameters = itk.ParameterObject.New()
        TRANS = ['translation', 'rigid', 'similarity', 'affine', 'bspline', 'spline', 'groupwise']

        trans = itkregist.str2list(trans, values=['rigid', 'bspline'])
        resolutions = itkregist.str2list(resolutions, values=[15]*len(trans))
        GridSpacing = itkregist.str2list(GridSpacing, values=[15]*len(trans))

        # if (set(trans) - set(TRANS)):
        #     raise TypeError(f'The valid transform type are {TRANS}.')

        for i, itran in enumerate(trans):
            ires = resolutions[i]
            igrid= GridSpacing[i]
            if  itran== 'similarity':
                default_para = parameters.GetDefaultParameterMap('rigid', ires, igrid)
                parameters.AddParameterMap(default_para)
                parameters.SetParameter(i, "Transform", "SimilarityTransform")
            else:
                try:
                    default_para = parameters.GetDefaultParameterMap(itran, ires, igrid)
                    parameters.AddParameterMap(default_para)
                except:
                    print(f'Cannot set {itran}  as the valid transtype! Will be replaced by translation.')
                    default_para = parameters.GetDefaultParameterMap('translation', ires, igrid)
                    parameters.AddParameterMap(default_para)
            
        for itr, itran in enumerate(trans):
            # parameters.SetParameter(itr, "Optimizer", "RegularStepGradientDescent")
            parameters.SetParameter(itr, "Optimizer", "AdaptiveStochasticGradientDescent")
            parameters.SetParameter(itr, "MaximumNumberOfIterations", "1000")
            parameters.SetParameter(itr, "MaximumStepLength", "1")
            parameters.SetParameter(itr, "NumberOfSpatialSamples", "3000")

            parameters.SetParameter(itr, "MovingImagePyramid", "FixedRecursiveImagePyramid")
            parameters.SetParameter(itr, "MovingImagePyramid", "MovingRecursiveImagePyramid")

            #parameters.SetParameter(itr, "NumberOfHistogramBins", ["16", "32" ,"64"])
            # parameters.SetParameter(itr, "NumberOfResolutions", "5")
            # parameters.SetParameter(0, "ImagePyramidSchedule",list(np.repeat([32, 16, 8, 4, 1], 2).astype(str)))
            # parameters.SetParameter(0, "FixedImagePyramidRescaleSchedule",list(np.repeat([64, 32, 16, 8, 4, 1], 2).astype(str)))
            # parameters.SetParameter(0, "MovingImagePyramidRescaleSchedule",list(np.repeat([64, 32, 16, 8, 4, 1], 2).astype(str)))

            # parameters.SetParameter(itr, "Metric", "AdvancedMattesMutualInformation")

            # parameters.SetParameter(itr, "UseDirectionCosines", "true")
            # parameters.SetParameter(itr, "FixedInternalImagePixelType", "float")
            # parameters.SetParameter(itr, "MovingInternalImagePixelType", "float")
            # parameters.SetParameter(itr, "AutomaticTransformInitialization", "true")
            # parameters.SetParameter(itr, "AutomaticTransformInitializationMethod", "GeometricCenter")
            # parameters.SetParameter(itr, "WriteResultImage ", "false")
        #parameters.RemoveParameter("ResultImageFormat")
        if verb:
            print(parameters)
        return parameters