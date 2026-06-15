import os
import SimpleITK as sitk


def load_dicom_series(folder):
    """
    读取一个 DICOM 序列为 SimpleITK Image。
    如果文件夹中有多个 Series，会默认读取切片数最多的那个。
    """

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {folder}")

    best_series_id = None
    best_file_names = None
    max_num_files = 0

    print(f"\nFound {len(series_ids)} series in {folder}")

    for sid in series_ids:
        file_names = reader.GetGDCMSeriesFileNames(folder, sid)
        print(f"Series: {sid}, number of files: {len(file_names)}")

        if len(file_names) > max_num_files:
            max_num_files = len(file_names)
            best_series_id = sid
            best_file_names = file_names

    print(f"Use series: {best_series_id}")

    reader.SetFileNames(best_file_names)
    image = reader.Execute()

    return image


def print_image_info(name, image):
    print(f"\n{name}")
    print("Dimension:", image.GetDimension())
    print("Pixel type:", image.GetPixelIDTypeAsString())
    print("Size:", image.GetSize())
    print("Spacing:", image.GetSpacing())
    print("Origin:", image.GetOrigin())
    print("Direction:", image.GetDirection())


def clamp_and_normalize(image, lower, upper):
    """
    CT 强度截断 + 归一化到 0~1，输出强制为 Float32。
    """

    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Clamp(image, sitk.sitkFloat32, lower, upper)
    image = sitk.IntensityWindowing(
        image,
        windowMinimum=lower,
        windowMaximum=upper,
        outputMinimum=0.0,
        outputMaximum=1.0
    )
    image = sitk.Cast(image, sitk.sitkFloat32)

    return image

def normalize_mri_by_statistics(image):
    """
    MRI 简单归一化，输出强制为 Float32。
    """

    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Normalize(image)
    image = sitk.RescaleIntensity(image, 0.0, 1.0)
    image = sitk.Cast(image, sitk.sitkFloat32)

    return image


def resample_to_spacing(
    image,
    new_spacing,
    interpolator=sitk.sitkLinear,
    default_value=0,
    output_pixel_type=sitk.sitkFloat32
):
    """
    将图像重采样到指定 spacing。
    输出强制为 Float32，避免 SimpleITK 配准时 fixed/moving 类型不一致。
    """

    image = sitk.Cast(image, output_pixel_type)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if image.GetDimension() != 3:
        raise RuntimeError(
            f"Expected 3D image, but got dimension={image.GetDimension()}"
        )

    new_size = [
        max(1, int(round(original_size[i] * original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_value,
        output_pixel_type
    )

    return resampled

def rigid_register_ct_mr_sag(ct_img, mr_img, output_transform_path):
    """
    fixed = CT
    moving = MR_SAG

    返回的 final_transform 可直接用于：
    sitk.Resample(mr_img, ct_img, final_transform, ...)
    即把 MRI 重采样到 CT 空间。
    """

    # ---------- 1. 强度预处理 ----------
    # CT：保留骨和软组织范围。腰椎 CT 可先用 [-300, 1200]
    fixed = clamp_and_normalize(ct_img, lower=-300, upper=1200)

    # MRI：简单归一化
    moving = normalize_mri_by_statistics(mr_img)

    # ---------- 2. 降采样加快配准 ----------
    # 注意：MR_SAG 第三维 spacing 是 4.4 mm，所以不要强行降到 1 mm。
    fixed_low = resample_to_spacing(
        fixed,
        new_spacing=(2.0, 2.0, 2.0),
        interpolator=sitk.sitkLinear,
        default_value=0
    )

    moving_low = resample_to_spacing(
        moving,
        new_spacing=(2.0, 2.0, 2.0),
        interpolator=sitk.sitkLinear,
        default_value=0
    )

    # 关键修复：强制统一像素类型
    fixed_low = sitk.Cast(fixed_low, sitk.sitkFloat32)
    moving_low = sitk.Cast(moving_low, sitk.sitkFloat32)

    print_image_info("Fixed low CT", fixed_low)
    print_image_info("Moving low MR_SAG", moving_low)

    if fixed_low.GetDimension() != moving_low.GetDimension():
        raise RuntimeError(
            f"Dimension mismatch: fixed={fixed_low.GetDimension()}, "
            f"moving={moving_low.GetDimension()}"
        )

    if fixed_low.GetDimension() != 3:
        raise RuntimeError(
            f"Images must be 3D, but got fixed dimension={fixed_low.GetDimension()}"
        )

    # ---------- 3. 中心初始化 ----------
    # 这一步相当于自动给一个初始位姿。
    # 因为 CT 和 MRI FrameOfReferenceUID 不同，不能用原始 origin 直接对齐。
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_low,
        moving_low,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    print("\nInitial transform:")
    print(initial_transform)

    # ---------- 4. 刚性配准 ----------
    registration_method = sitk.ImageRegistrationMethod()

    # 多模态图像推荐 Mattes Mutual Information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # MRI 体积层数少，不要采样太少。
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.20)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # 优化器
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 多分辨率配准
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 监控迭代
    def command_iteration(method):
        print(
            f"Iteration: {method.GetOptimizerIteration():4d} | "
            f"Metric: {method.GetMetricValue():.6f} | "
            f"LR: {method.GetOptimizerLearningRate():.6f}"
        )

    registration_method.AddCommand(
        sitk.sitkIterationEvent,
        lambda: command_iteration(registration_method)
    )

    print("\nStart rigid registration...")
    final_transform_low = registration_method.Execute(fixed_low, moving_low)

    print("\nRegistration finished.")
    print("Final metric value:", registration_method.GetMetricValue())
    print("Optimizer stop condition:")
    print(registration_method.GetOptimizerStopConditionDescription())

    # ---------- 5. 用低分辨率得到的 transform 直接作用于原始图像 ----------
    # 因为 SimpleITK transform 是物理空间变换，低分辨率/原始分辨率通用。
    sitk.WriteTransform(final_transform_low, output_transform_path)

    print(f"\nSaved transform to: {output_transform_path}")

    return final_transform_low


def resample_mri_to_ct(mr_img, ct_img, transform_ct_to_mr, output_path):
    """
    把 MR_SAG 重采样到 CT 空间。
    输出大小、spacing、origin、direction 与 CT 一致。
    """

    mr_to_ct = sitk.Resample(
        mr_img,
        ct_img,
        transform_ct_to_mr,
        sitk.sitkLinear,
        0,
        sitk.sitkFloat32
    )

    sitk.WriteImage(mr_to_ct, output_path)
    print(f"Saved MRI resampled to CT: {output_path}")

    return mr_to_ct


def create_mri_valid_mask_in_ct(mr_img, ct_img, transform_ct_to_mr, output_path):
    """
    生成 MRI 在 CT 空间中的有效覆盖区域。
    """

    mask = sitk.Image(mr_img.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(mr_img)
    mask = mask + 1

    mask_ct = sitk.Resample(
        mask,
        ct_img,
        transform_ct_to_mr,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    sitk.WriteImage(mask_ct, output_path)
    print(f"Saved MRI valid mask in CT: {output_path}")

    return mask_ct


def resample_ct_to_mri_planes(ct_img, mr_img, transform_ct_to_mr, output_path):
    """
    从 CT 中提取与 MR_SAG 每一层完全对应的 CT 斜切面。

    输出结果与 MR_SAG 的 size、spacing、origin、direction 一致。
    输出第 k 层 = MR_SAG 第 k 层在 CT 中对应的 CT 图像。
    """

    transform_mr_to_ct = transform_ct_to_mr.GetInverse()

    ct_on_mri_planes = sitk.Resample(
        ct_img,
        mr_img,
        transform_mr_to_ct,
        sitk.sitkLinear,
        -1024,
        sitk.sitkFloat32
    )

    sitk.WriteImage(ct_on_mri_planes, output_path)
    print(f"Saved CT matched to MRI planes: {output_path}")

    return ct_on_mri_planes


if __name__ == "__main__":

    ct_folder = r"F:\vitual scalpel\image registration\data\jingzhui\CT_jingzhui"
    mr_sag_folder = r"F:\vitual scalpel\image registration\data\jingzhui\MR_jingzhui"

    output_dir = r"F:\vitual scalpel\image registration\data\jingzhui\results"
    os.makedirs(output_dir, exist_ok=True)

    transform_path = os.path.join(output_dir, "CT_to_MR_SAG_rigid.tfm")

    ct_img = load_dicom_series(ct_folder)
    mr_sag_img = load_dicom_series(mr_sag_folder)

    print_image_info("Original CT", ct_img)
    print_image_info("Original MR_SAG", mr_sag_img)

    # 保存从 DICOM 读取出来的原始 NIfTI，方便在 ITK-SNAP / 其他工具中查看
    sitk.WriteImage(ct_img, os.path.join(output_dir, "CT_from_DICOM.nii.gz"))
    sitk.WriteImage(mr_sag_img, os.path.join(output_dir, "MR_from_DICOM.nii.gz"))

    # 1. 刚性配准
    final_transform = rigid_register_ct_mr_sag(
        ct_img,
        mr_sag_img,
        transform_path
    )

    # 2. MRI 重采样到 CT 空间
    resample_mri_to_ct(
        mr_sag_img,
        ct_img,   
        final_transform,
        os.path.join(output_dir, "MR_resampled_to_CT.nii.gz")
    )

    # 3. MRI 有效区域 mask
    create_mri_valid_mask_in_ct(
        mr_sag_img,
        ct_img,
        final_transform,
        os.path.join(output_dir, "MR_valid_mask_in_CT.nii.gz")
    )

    # 4. 提取 CT 中与 MR_SAG 每层对应的斜切面
    resample_ct_to_mri_planes(
        ct_img,
        mr_sag_img,
        final_transform,
        os.path.join(output_dir, "CT_matched_to_MR.nii.gz")
    )

    print("\nAll done.")