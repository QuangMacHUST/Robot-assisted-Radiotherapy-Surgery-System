#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboRadAssist - Module xử lý hình ảnh y tế
Cung cấp các chức năng xử lý hình ảnh DICOM, phân đoạn và tái tạo 3D
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union
import pydicom
from pathlib import Path

# Import AI modules (sẽ được phát triển sau)
# from src.ai.segmentation import segmentation_models
# from src.ai.registration import registration_models

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Class quản lý và xử lý hình ảnh y tế
    """
    
    def __init__(self):
        """Khởi tạo Image Processor"""
        self.current_study = None
        self.loaded_series = {}
        self.current_segmentation = None
        self.registration_results = {}
        self.cache_dir = Path("./data/cache")
        
    def initialize(self):
        """Khởi tạo hệ thống xử lý hình ảnh"""
        logger.info("Initializing medical image processing system")
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Load các mô hình AI (nếu cần)
            logger.info("Medical image processing system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize medical image processing system: {e}")
            return False
            
    def load_dicom_series(self, directory_path: Union[str, Path], series_id: Optional[str] = None) -> Dict:
        """
        Tải dữ liệu từ thư mục DICOM
        
        Args:
            directory_path: Đường dẫn đến thư mục chứa file DICOM
            series_id: ID của series cần tải (nếu None, tải tất cả series)
            
        Returns:
            Dict: Thông tin về các series đã tải
        """
        try:
            directory_path = Path(directory_path)
            logger.info(f"Loading DICOM series from {directory_path}")
            
            reader = sitk.ImageSeriesReader()
            
            if series_id is None:
                # Tải tất cả series trong thư mục
                series_IDs = reader.GetGDCMSeriesIDs(str(directory_path))
                
                if not series_IDs:
                    logger.warning(f"No DICOM series found in {directory_path}")
                    return {}
                
                result = {}
                for series_id in series_IDs:
                    dicom_names = reader.GetGDCMSeriesFileNames(str(directory_path), series_id)
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    
                    # Đọc metadata từ file đầu tiên
                    metadata = self._extract_metadata(dicom_names[0])
                    
                    self.loaded_series[series_id] = {
                        'image': image,
                        'filenames': dicom_names,
                        'metadata': metadata
                    }
                    
                    result[series_id] = {
                        'modality': metadata.get('Modality', 'Unknown'),
                        'description': metadata.get('SeriesDescription', 'Unknown'),
                        'date': metadata.get('StudyDate', 'Unknown'),
                        'slice_count': len(dicom_names)
                    }
                
                return result
            else:
                # Tải chỉ một series cụ thể
                dicom_names = reader.GetGDCMSeriesFileNames(str(directory_path), series_id)
                
                if not dicom_names:
                    logger.warning(f"Series {series_id} not found in {directory_path}")
                    return {}
                
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                
                # Đọc metadata từ file đầu tiên
                metadata = self._extract_metadata(dicom_names[0])
                
                self.loaded_series[series_id] = {
                    'image': image,
                    'filenames': dicom_names,
                    'metadata': metadata
                }
                
                return {
                    series_id: {
                        'modality': metadata.get('Modality', 'Unknown'),
                        'description': metadata.get('SeriesDescription', 'Unknown'),
                        'date': metadata.get('StudyDate', 'Unknown'),
                        'slice_count': len(dicom_names)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error loading DICOM series: {e}")
            return {}
    
    def _extract_metadata(self, dicom_file: str) -> Dict:
        """
        Trích xuất metadata từ file DICOM
        
        Args:
            dicom_file: Đường dẫn đến file DICOM
            
        Returns:
            Dict: Metadata trích xuất được
        """
        try:
            dataset = pydicom.dcmread(dicom_file)
            metadata = {
                'PatientName': str(dataset.PatientName) if hasattr(dataset, 'PatientName') else 'Unknown',
                'PatientID': dataset.PatientID if hasattr(dataset, 'PatientID') else 'Unknown',
                'StudyDate': dataset.StudyDate if hasattr(dataset, 'StudyDate') else 'Unknown',
                'Modality': dataset.Modality if hasattr(dataset, 'Modality') else 'Unknown',
                'SeriesDescription': dataset.SeriesDescription if hasattr(dataset, 'SeriesDescription') else 'Unknown',
                'SliceThickness': dataset.SliceThickness if hasattr(dataset, 'SliceThickness') else 0,
                'PixelSpacing': dataset.PixelSpacing if hasattr(dataset, 'PixelSpacing') else [1, 1],
                'ImageOrientationPatient': dataset.ImageOrientationPatient if hasattr(dataset, 'ImageOrientationPatient') else [1, 0, 0, 0, 1, 0],
                'ImagePositionPatient': dataset.ImagePositionPatient if hasattr(dataset, 'ImagePositionPatient') else [0, 0, 0]
            }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting DICOM metadata: {e}")
            return {}
    
    def segment_structures(self, series_id: str, structures: List[str] = ['tumor', 'organs_at_risk']) -> Dict:
        """
        Phân đoạn các cấu trúc từ hình ảnh
        
        Args:
            series_id: ID của series để phân đoạn
            structures: Danh sách các cấu trúc cần phân đoạn
            
        Returns:
            Dict: Kết quả phân đoạn, key là tên cấu trúc, value là mask
        """
        if series_id not in self.loaded_series:
            logger.error(f"Series {series_id} not loaded")
            return {}
            
        try:
            logger.info(f"Segmenting structures: {structures} from series {series_id}")
            
            # Lấy hình ảnh để phân đoạn
            image_data = self.loaded_series[series_id]['image']
            
            # Chuyển đổi SimpleITK image thành mảng NumPy để xử lý
            np_image = sitk.GetArrayFromImage(image_data)
            
            # Trong phiên bản thực tế, đây sẽ gọi đến mô hình AI phân đoạn
            # Ở đây, tạo ra kết quả mô phỏng
            segmentation_results = {}
            
            # Mô phỏng kết quả phân đoạn đơn giản
            for structure in structures:
                if structure == 'tumor':
                    # Mô phỏng một khối u ở giữa hình ảnh
                    mask = np.zeros_like(np_image)
                    center = np.array(mask.shape) // 2
                    radius = min(mask.shape) // 10
                    
                    # Tạo khối u hình cầu đơn giản
                    for z in range(mask.shape[0]):
                        for y in range(mask.shape[1]):
                            for x in range(mask.shape[2]):
                                if np.sum(((z, y, x) - center)**2) < radius**2:
                                    mask[z, y, x] = 1
                                    
                    # Chuyển đổi mask thành SimpleITK image
                    tumor_mask = sitk.GetImageFromArray(mask)
                    tumor_mask.CopyInformation(image_data)  # Sao chép thông tin không gian
                    
                    segmentation_results['tumor'] = tumor_mask
                
                elif structure == 'organs_at_risk':
                    # Mô phỏng các cơ quan nguy cơ đơn giản
                    mask = np.zeros_like(np_image)
                    center = np.array(mask.shape) // 2
                    radius = min(mask.shape) // 5
                    
                    # Tạo cơ quan đơn giản
                    for z in range(mask.shape[0]):
                        for y in range(mask.shape[1]):
                            for x in range(mask.shape[2]):
                                dist = np.sum(((z, y, x) - center)**2)
                                if radius**2 < dist < (radius*1.5)**2:
                                    mask[z, y, x] = 1
                    
                    # Chuyển đổi mask thành SimpleITK image
                    oar_mask = sitk.GetImageFromArray(mask)
                    oar_mask.CopyInformation(image_data)  # Sao chép thông tin không gian
                    
                    segmentation_results['organs_at_risk'] = oar_mask
            
            self.current_segmentation = segmentation_results
            
            logger.info(f"Segmentation completed successfully")
            return {k: sitk.GetArrayFromImage(v).sum() for k, v in segmentation_results.items()}
            
        except Exception as e:
            logger.error(f"Error segmenting structures: {e}")
            return {}
    
    def register_images(self, fixed_series_id: str, moving_series_id: str, transform_type: str = 'rigid') -> bool:
        """
        Đăng ký hai hình ảnh với nhau
        
        Args:
            fixed_series_id: ID của series cố định
            moving_series_id: ID của series di chuyển
            transform_type: Loại chuyển đổi ('rigid', 'affine', 'deformable')
            
        Returns:
            bool: True nếu đăng ký thành công
        """
        if fixed_series_id not in self.loaded_series or moving_series_id not in self.loaded_series:
            logger.error(f"One or both series not loaded")
            return False
            
        try:
            logger.info(f"Registering series {moving_series_id} to {fixed_series_id} using {transform_type} transform")
            
            fixed_image = self.loaded_series[fixed_series_id]['image']
            moving_image = self.loaded_series[moving_series_id]['image']
            
            # Thực hiện đăng ký hình ảnh
            if transform_type == 'rigid':
                registration_method = sitk.ImageRegistrationMethod()
                
                # Thiết lập tối ưu hóa
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                registration_method.SetMetricSamplingPercentage(0.01)
                registration_method.SetInterpolator(sitk.sitkLinear)
                
                # Thiết lập tối ưu hóa
                registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
                registration_method.SetOptimizerScalesFromPhysicalShift()
                
                # Thiết lập chuyển đổi
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image, 
                    moving_image, 
                    sitk.Euler3DTransform(), 
                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
                
                registration_method.SetInitialTransform(initial_transform, inPlace=False)
                
                # Thực hiện đăng ký
                final_transform = registration_method.Execute(fixed_image, moving_image)
                
                # Áp dụng chuyển đổi
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(fixed_image)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(0)
                resampler.SetTransform(final_transform)
                
                registered_image = resampler.Execute(moving_image)
                
                # Lưu kết quả
                self.registration_results[(fixed_series_id, moving_series_id)] = {
                    'transform': final_transform,
                    'registered_image': registered_image,
                    'transform_type': transform_type
                }
                
                logger.info(f"Registration completed successfully")
                return True
                
            elif transform_type == 'affine':
                # Tương tự như trên nhưng sử dụng AffineTransform
                logger.info("Affine registration not implemented in this version")
                return False
                
            elif transform_type == 'deformable':
                # Đăng ký biến dạng phức tạp hơn
                logger.info("Deformable registration not implemented in this version")
                return False
                
            else:
                logger.error(f"Unknown transform type: {transform_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering images: {e}")
            return False
    
    def create_3d_model(self, series_id: str, structure_name: Optional[str] = None) -> Dict:
        """
        Tạo mô hình 3D từ hình ảnh hoặc kết quả phân đoạn
        
        Args:
            series_id: ID của series để tạo mô hình
            structure_name: Tên của cấu trúc để tạo mô hình (nếu None, sử dụng toàn bộ hình ảnh)
            
        Returns:
            Dict: Thông tin về mô hình 3D đã tạo
        """
        if series_id not in self.loaded_series:
            logger.error(f"Series {series_id} not loaded")
            return {}
            
        try:
            logger.info(f"Creating 3D model for series {series_id}" + 
                      (f", structure: {structure_name}" if structure_name else ""))
            
            if structure_name is None:
                # Tạo mô hình từ hình ảnh gốc
                image = self.loaded_series[series_id]['image']
                
                # Thực hiện thuật toán Marching Cubes
                # Trong phiên bản thực tế, đây sẽ sử dụng VTK hoặc các thư viện tương tự
                
                # Mô phỏng kết quả
                model_info = {
                    'series_id': series_id,
                    'structure_name': 'volume',
                    'vertices_count': 10000,  # Mô phỏng
                    'faces_count': 20000,     # Mô phỏng
                    'model_path': str(self.cache_dir / f"{series_id}_volume_model.obj")
                }
                
                logger.info(f"3D model created successfully")
                return model_info
                
            else:
                # Kiểm tra xem đã phân đoạn chưa
                if self.current_segmentation is None or structure_name not in self.current_segmentation:
                    logger.error(f"Structure {structure_name} not segmented")
                    return {}
                
                # Lấy mask cấu trúc
                structure_mask = self.current_segmentation[structure_name]
                
                # Thực hiện thuật toán Marching Cubes
                # Trong phiên bản thực tế, đây sẽ sử dụng VTK hoặc các thư viện tương tự
                
                # Mô phỏng kết quả
                model_info = {
                    'series_id': series_id,
                    'structure_name': structure_name,
                    'vertices_count': 5000,  # Mô phỏng
                    'faces_count': 10000,    # Mô phỏng
                    'model_path': str(self.cache_dir / f"{series_id}_{structure_name}_model.obj")
                }
                
                logger.info(f"3D model created successfully for structure {structure_name}")
                return model_info
                
        except Exception as e:
            logger.error(f"Error creating 3D model: {e}")
            return {}
    
    def calculate_volume(self, structure_name: str) -> float:
        """
        Tính toán thể tích của một cấu trúc đã phân đoạn
        
        Args:
            structure_name: Tên của cấu trúc để tính thể tích
            
        Returns:
            float: Thể tích của cấu trúc (ml)
        """
        if self.current_segmentation is None or structure_name not in self.current_segmentation:
            logger.error(f"Structure {structure_name} not segmented")
            return 0.0
            
        try:
            # Lấy mask cấu trúc
            structure_mask = self.current_segmentation[structure_name]
            
            # Lấy kích thước voxel
            spacing = structure_mask.GetSpacing()
            
            # Lấy dữ liệu mask dưới dạng NumPy array
            mask_array = sitk.GetArrayFromImage(structure_mask)
            
            # Tính thể tích
            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3
            total_volume = np.sum(mask_array) * voxel_volume / 1000.0  # convert to ml
            
            logger.info(f"Volume of {structure_name}: {total_volume} ml")
            return total_volume
            
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return 0.0
    
    def export_dicom_rt(self, output_path: Union[str, Path], structures: List[str]) -> bool:
        """
        Xuất kết quả phân đoạn sang định dạng DICOM RT
        
        Args:
            output_path: Đường dẫn đến thư mục đầu ra
            structures: Danh sách các cấu trúc để xuất
            
        Returns:
            bool: True nếu xuất thành công
        """
        if self.current_segmentation is None:
            logger.error("No segmentation results available")
            return False
            
        try:
            output_path = Path(output_path)
            os.makedirs(output_path, exist_ok=True)
            
            logger.info(f"Exporting DICOM RT structures to {output_path}")
            
            # Trong phiên bản thực tế, đây sẽ sử dụng thư viện như pydicom để tạo DICOM RT Structure Set
            # Ở đây, chúng ta mô phỏng việc xuất file
            
            rt_file_path = output_path / "rtstructure.dcm"
            
            # Mô phỏng việc lưu file
            with open(rt_file_path, 'w') as f:
                f.write("DICOM RT Structure Set - Mockup")
                
            logger.info(f"Exported DICOM RT to {rt_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting DICOM RT: {e}")
            return False
    
    def clear_segmentation(self):
        """Xóa kết quả phân đoạn hiện tại"""
        self.current_segmentation = None
        logger.info("Segmentation results cleared")
    
    def clear_loaded_series(self):
        """Xóa tất cả series đã tải"""
        self.loaded_series = {}
        self.current_segmentation = None
        self.registration_results = {}
        logger.info("All loaded series cleared")

# Singleton instance
_instance = None

def get_instance() -> ImageProcessor:
    """
    Lấy instance của ImageProcessor (Singleton pattern)
    
    Returns:
        ImageProcessor: Instance của image processor
    """
    global _instance
    if _instance is None:
        _instance = ImageProcessor()
    return _instance

def initialize():
    """Khởi tạo hệ thống xử lý hình ảnh"""
    return get_instance().initialize()
