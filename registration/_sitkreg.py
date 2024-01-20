import numpy as np
try:
    import SimpleITK as sitk
except:
    pass

class sitkregist:
    """
    This class is a wrapper for SimpleITK registration.
    """

    def __init__(self):
        pass

    @staticmethod
    def _transformer(transtype, dimension, fixed_image=None, 
                     grid_physical_spacing=[50,50,50], order =3 ):
        mesh_size = [
            int(size * spacing / grid_spacing + 0.5)
            for size, spacing, grid_spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing(), grid_physical_spacing)
        ]

        transdict = {
            2:{
                'translation': sitk.TranslationTransform(2),
                "euler": sitk.Euler2DTransform(),
                "rigid": sitk.Euler2DTransform(),
                'similarity': sitk.Similarity2DTransform(),
                'scale': sitk.ScaleTransform(2),
                'affine': sitk.AffineTransform(2),
                'bspline': sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize=mesh_size, order=order),
                # 'displacementfiled': sitk.DisplacementFieldTransform(2),
                # 'composite': sitk.CompositeTransform(2),
                # 'transform': sitk.Transform(2)
            },
            3:{
                'transform': sitk.TranslationTransform(3),
                'versor':sitk.VersorTransform(),
                'versorrigid':sitk.VersorRigid3DTransform(),
                "euler": sitk.Euler3DTransform(),
                "rigid": sitk.Euler3DTransform(),
                'similarity': sitk.Similarity3DTransform(),
                'scale': sitk.ScaleTransform(3),
                'scaleversor': sitk.ScaleVersor3DTransform(),
                'scaleskewversor': sitk.ScaleSkewVersor3DTransform(),
                'affine': sitk.AffineTransform(3),
                'bspline': sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize=mesh_size, order=order),
                # 'displacementfiled': sitk.DisplacementFieldTransform(3),
                # 'composite': sitk.CompositeTransform(3),
                # 'transform': sitk.Transform(3)
            },
        }
        try:
            tranformer = transdict[dimension][transtype]
        except KeyError:
            print('local difined transformer.')
            tranformer = transtype
        return tranformer

    @staticmethod
    def _Matrix(registration_method, matrix_type='information', 
                numberOfHistogramBins=None, radius=None, 
                varianceForJointPDFSmoothing=None):
        if matrix_type == 'information':
            registration_method.SetMetricAsMattesMutualInformation(
                                numberOfHistogramBins=numberOfHistogramBins or 50)
        elif matrix_type == 'ANTS':
            registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=radius or 50)
        elif matrix_type == 'correlation':
            registration_method.SetMetricAsCorrelation()
        elif matrix_type == 'Hinformation':
            registration_method.SetMetricAsJointHistogramMutualInformation(
                numberOfHistogramBins=numberOfHistogramBins or 20,
                varianceForJointPDFSmoothing=varianceForJointPDFSmoothing or 1.5)
        elif matrix_type == 'MeanSquares':
            registration_method.SetMetricAsMeanSquares()
        else:
            raise ValueError('Incorrect Matrix input')
        return registration_method

    @staticmethod
    def _optimizer(registration_method, optimizer_type='AsGradient', **kargs):
        if optimizer_type == 'AsGradient':
            registration_method.SetOptimizerAsGradientDescent(
                        learningRate=kargs.pop('learningRate',1.0),
                        numberOfIterations=kargs.pop('numberOfIterations',1000),
                        convergenceMinimumValue=kargs.pop('convergenceMinimumValue',10-6),
                        convergenceWindowSize=kargs.pop('convergenceWindowSize',10),
                        **kargs )
        elif optimizer_type == 'StepGradient':
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                kargs.pop('learningRate',1.0),
                kargs.pop('minStep',1e-4),
                kargs.pop('numberOfIterations',1000),
                relaxationFactor=kargs.pop('relaxationFactor',0.5),
                gradientMagnitudeTolerance=kargs.pop('gradientMagnitudeTolerance', 1e-4),
                maximumStepSizeInPhysicalUnits=kargs.pop('maximumStepSizeInPhysicalUnits',0.0),
                **kargs)
        elif optimizer_type == 'LBFGSB':
            registration_method.SetOptimizerAsLBFGSB(
                    gradientConvergenceTolerance=kargs.pop('gradientConvergenceTolerance', 1e-5),
                    numberOfIterations=kargs.pop('numberOfIterations',1000),
                    maximumNumberOfCorrections=kargs.pop('maximumNumberOfCorrections',5),
                    maximumNumberOfFunctionEvaluations=kargs.pop('maximumNumberOfFunctionEvaluations', 2000),
                    costFunctionConvergenceFactor=kargs.pop('costFunctionConvergenceFactor',1e+7),
                **kargs)
        elif optimizer_type == 'LBFGS2':
            registration_method.SetOptimizerAsLBFGS2(
                    numberOfIterations= kargs.pop('numberOfIterations',0),
                    hessianApproximateAccuracy= kargs.pop('hessianApproximateAccuracy',6),
                    deltaConvergenceDistance= kargs.pop('deltaConvergenceDistance',0),
                    deltaConvergenceTolerance= kargs.pop('deltaConvergenceTolerance',1e-5),
                    lineSearchMaximumEvaluations= kargs.pop('lineSearchMaximumEvaluations',40),
                    lineSearchMinimumStep= kargs.pop('lineSearchMinimumStep',1e-20),
                    lineSearchMaximumStep= kargs.pop('lineSearchMaximumStep',1e20),
                    lineSearchAccuracy= kargs.pop('lineSearchAccuracy',1e-4),
                **kargs)
        elif optimizer_type == 'Exhaustive':
            registration_method.SetOptimizerAsExhaustive(
                    kargs.pop('numberOfSteps',  [2,2,2,1,1,1]),
                    stepLength= kargs.pop('stepLength',1.0),
                **kargs)
        elif optimizer_type == 'Amoeba':
            registration_method.SetOptimizerAsAmoeba(
                        kargs.pop('simplexDelta', 2),
                        kargs.pop('numberOfIterations', 1000),
                        parametersConvergenceTolerance= kargs.pop('parametersConvergenceTolerance',1e-8),
                        functionConvergenceTolerance= kargs.pop('functionConvergenceTolerance',1e-4),
                **kargs)
        elif optimizer_type == 'Weights':
            registration_method.SetOptimizerWeights( kargs.pop('weights',[1,1,1,1,1,1]) )
            #Euler3DTransform:[angleX, angleY, angleZ, tx, ty, tz]
        elif optimizer_type == 'Powell':
            registration_method.SetOptimizerAsPowell(
                    numberOfIterations= kargs.pop('maximumStepSizeInPhysicalUnits',100),
                    maximumLineIterations= kargs.pop('maximumStepSizeInPhysicalUnits',100),
                    stepLength= kargs.pop('maximumStepSizeInPhysicalUnits',1),
                    stepTolerance= kargs.pop('maximumStepSizeInPhysicalUnits',1e-6),
                    valueTolerance= kargs.pop('maximumStepSizeInPhysicalUnits',1e-6),
                **kargs)
        else:
            raise ValueError('Incorrect optimizer input')
        return registration_method
    
    @staticmethod
    def _interpolator(itp_type='linear'):
        return {
                "linear": sitk.sitkLinear,
                "nearest": sitk.sitkNearestNeighbor,
                "BSpline": sitk.sitkBSpline,
            }.get(itp_type, 'Incorrect interpolator input')

    def regist(self, 
                fixed_image,
                moving_image,
                dimension=None,
                transtype = 'rigid',
                matrix_type='information', 
                optimizer_type='AsGradient',
                itp_type='linear',
                matrix_kargs = {},
                optimizer_kargs = {},

                msp= 0.05,
                centralRegionRadius=5, 
                smallParameterVariation=0.01,
                fixed_image_mask=None,
                number_of_threads = 64,

                shrinkFactors=[4, 2, 1],
                smoothingSigmas=[2, 1, 0],
                AddCommand = None,
                verbose=1,
                debugon = False,
                **kargs):

        if not isinstance(fixed_image, sitk.Image):
            fixed_image = sitk.GetImageFromArray(fixed_image.astype(np.float32))
            moving_image = sitk.GetImageFromArray(moving_image.astype(np.float32))

        dimension = dimension or fixed_image.GetDimension()
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image,
            self._transformer(transtype, dimension, fixed_image=fixed_image),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            initial_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID())

        registration_method = sitk.ImageRegistrationMethod()
        # Similarity metric settings.

        self._Matrix(registration_method, 
                        matrix_type=matrix_type, 
                        **matrix_kargs)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(msp)

        # Interpolator settings.
        registration_method.SetInterpolator(self._interpolator(itp_type))
        # Optimizer settings.
        registration_method.SetNumberOfThreads(number_of_threads)
        self._optimizer(registration_method, 
                        optimizer_type=optimizer_type,
                        **optimizer_kargs)

        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetOptimizerScalesFromPhysicalShift(
            centralRegionRadius=centralRegionRadius, 
            smallParameterVariation=smallParameterVariation)

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactors)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell
        if not fixed_image_mask is None:
            registration_method.SetMetricFixedMask(fixed_image_mask)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        if debugon:
            registration_method.DebugOn()

        if not AddCommand is None:
            for icommd in AddCommand:
                if isinstance(icommd, str):
                    registration_method.AddCommand(*eval(icommd))
                else:
                    registration_method.AddCommand(*icommd)
 
        verbose and print(registration_method)
        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(moving_image, sitk.sitkFloat32))

        transformed_moving = sitk.TransformGeometry(moving_image, final_transform)

        # Query the registration method to see the metric value and the reason the
        # optimization terminated.
        print('Final metric value: {0}'.format(
            registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(
            registration_method.GetOptimizerStopConditionDescription()))

        self.registration = registration_method
        return final_transform, transformed_moving

        # Optimizer settings.
        # https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/62_Registration_Tuning.html
        ## GDS
        # registration_method.SetOptimizerAsGradientDescent(
        #     learningRate=0.1,
        #     numberOfIterations=1000,
        #     convergenceMinimumValue=1e-4,
        #     convergenceWindowSize=10,
        # )
        # registration_method.SetOptimizerScalesFromPhysicalShift()
        ##Exhaustive

        # registration_method.SetOptimizerAsExhaustive(
        #     numberOfSteps=[10,10,10,10,10,10], stepLength=np.pi/2
        # )
        # registration_method.SetOptimizerScales([2, 2, 2, 2, 2, 2])
        # sample_per_axis = 12
        # registration_method.SetOptimizerAsExhaustive(
        #         [
        #             sample_per_axis // 2,
        #             sample_per_axis // 2,
        #             sample_per_axis // 4,
        #             0,
        #             0,
        #             0,
        #         ]
        #     )
        # registration_method.SetOptimizerScales(
        #         [
        #             2.0 * np.pi / sample_per_axis,
        #             2.0 * np.pi / sample_per_axis,
        #             2.0 * np.pi / sample_per_axis,
        #             1.0,
        #             1.0,
        #             1.0,
        #         ]
        #     )

        # Setup for the multi-resolution framework.
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[6,4,2,1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2,1,0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Connect all of the observers so that we can perform plotting during registration.
        # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        # registration_method.AddCommand(
        #     sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
        # )
        # registration_method.AddCommand(
        #     sitk.sitkIterationEvent, lambda: plot_values(registration_method)
        # )

        # final_transform = registration_method.Execute(
        #     sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
        # )
