#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboRadAssist - Module tính toán liều xạ trị
Cung cấp các chức năng tính toán và mô phỏng phân bố liều xạ trị
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class DoseCalculator:
    """Class tính toán và mô phỏng phân bố liều xạ trị"""
    
    def __init__(self):
        """Khởi tạo Dose Calculator"""
        self.beam_models = {}
        self.current_calculation = None
        self.calculation_in_progress = False
        self.mutex = threading.Lock()
        self.dose_cache = {}
        
    def initialize(self):
        """Khởi tạo hệ thống tính toán liều"""
        logger.info("Initializing dose calculation system")
        try:
            # Tải mô hình chùm tia
            self._load_beam_models()
            logger.info("Dose calculation system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize dose calculation system: {e}")
            return False
    
    def _load_beam_models(self):
        """Tải các mô hình chùm tia từ dữ liệu"""
        # Trong hệ thống thực, đây sẽ tải dữ liệu từ các file đo đạc
        # Ở đây, chúng ta tạo mô hình mô phỏng đơn giản
        
        # Mô hình chùm tia 6MV
        self.beam_models["6MV"] = {
            "PDD": self._generate_pdd(energy=6),
            "profiles": self._generate_profiles(energy=6),
            "output_factors": self._generate_output_factors(energy=6)
        }
        
        # Mô hình chùm tia 10MV
        self.beam_models["10MV"] = {
            "PDD": self._generate_pdd(energy=10),
            "profiles": self._generate_profiles(energy=10),
            "output_factors": self._generate_output_factors(energy=10)
        }
        
        logger.info(f"Loaded beam models for: {list(self.beam_models.keys())}")
    
    def _generate_pdd(self, energy: int) -> Dict:
        """
        Tạo dữ liệu PDD (Percentage Depth Dose) mô phỏng
        
        Args:
            energy: Năng lượng chùm tia (MV)
            
        Returns:
            Dict: Dữ liệu PDD
        """
        # Mô phỏng PDD cho chùm tia
        depths = np.arange(0, 30.1, 0.1)  # cm
        
        # Mô phỏng đơn giản dựa trên công thức
        d_max = 1.5 if energy == 6 else 2.5  # Độ sâu liều tối đa
        mu = 0.04 if energy == 6 else 0.03   # Hệ số suy giảm
        
        # Mô phỏng build-up và fall-off
        pdd = np.zeros_like(depths)
        for i, d in enumerate(depths):
            if d < d_max:
                # Build-up region
                pdd[i] = 100 * (d / d_max) ** 1.5
            else:
                # Fall-off region
                pdd[i] = 100 * np.exp(-mu * (d - d_max))
        
        return {
            "depths": depths.tolist(),
            "values": pdd.tolist(),
            "ssd": 100.0  # Source-Surface Distance (cm)
        }
    
    def _generate_profiles(self, energy: int) -> Dict:
        """
        Tạo dữ liệu profiles mô phỏng
        
        Args:
            energy: Năng lượng chùm tia (MV)
            
        Returns:
            Dict: Dữ liệu profiles
        """
        # Mô phỏng profiles cho chùm tia
        field_sizes = [5, 10, 15, 20]  # cm
        depths = [1.5, 5, 10, 20]  # cm
        
        profiles = {}
        
        for fs in field_sizes:
            profiles[f"{fs}x{fs}"] = {}
            
            for d in depths:
                # Tạo profile theo trục X
                x = np.arange(-25, 25.1, 0.5)  # cm
                
                # Mô phỏng đơn giản dựa trên hàm error function
                sigma = 0.35 * fs  # Độ rộng của penumbra phụ thuộc vào kích thước trường
                profile = np.zeros_like(x)
                
                for i, xi in enumerate(x):
                    if abs(xi) <= fs/2:
                        # Trong trường
                        profile[i] = 100
                    else:
                        # Penumbra và ngoài trường
                        dist = abs(xi) - fs/2
                        profile[i] = 100 * np.exp(-dist**2 / (2 * sigma**2))
                
                # Thêm nhiễu nhỏ
                profile += np.random.normal(0, 0.5, size=profile.shape)
                
                profiles[f"{fs}x{fs}"][f"d={d}"] = {
                    "x": x.tolist(),
                    "values": profile.tolist()
                }
        
        return profiles
    
    def _generate_output_factors(self, energy: int) -> Dict:
        """
        Tạo dữ liệu output factors mô phỏng
        
        Args:
            energy: Năng lượng chùm tia (MV)
            
        Returns:
            Dict: Dữ liệu output factors
        """
        # Mô phỏng output factors cho các kích thước trường
        field_sizes = np.arange(4, 40.1, 1)  # cm
        
        # Mô phỏng đơn giản
        of = 0.8 + 0.2 * (1 - np.exp(-0.1 * field_sizes))
        
        return {
            "field_sizes": field_sizes.tolist(),
            "values": of.tolist()
        }
    
    def calculate_point_dose(self, beam_params: Dict, point: List[float]) -> float:
        """
        Tính toán liều tại một điểm
        
        Args:
            beam_params: Thông số chùm tia
            point: Tọa độ điểm cần tính [x, y, z] (mm)
            
        Returns:
            float: Liều tại điểm (Gy)
        """
        try:
            # Lấy thông số chùm tia
            energy = beam_params.get("energy", "6MV")
            ssd = beam_params.get("ssd", 1000.0)  # mm
            field_size = beam_params.get("field_size", [100, 100])  # mm
            gantry_angle = beam_params.get("gantry_angle", 0)  # degrees
            collimator_angle = beam_params.get("collimator_angle", 0)  # degrees
            mu = beam_params.get("mu", 100.0)  # Monitor Units
            
            # Chuyển đổi sang cm cho tính toán
            point_cm = [p / 10.0 for p in point]
            field_size_cm = [f / 10.0 for f in field_size]
            ssd_cm = ssd / 10.0
            
            # Kiểm tra xem có mô hình chùm tia hay không
            if energy not in self.beam_models:
                logger.error(f"Beam model for energy {energy} not found")
                return 0.0
            
            # Mô phỏng tính toán liều đơn giản
            # 1. Tính khoảng cách từ nguồn đến điểm
            # Để đơn giản, giả sử điểm nằm trên trục trung tâm
            depth = point_cm[2]  # Độ sâu (giả sử theo trục z)
            sad = ssd_cm + depth  # Source-Axis Distance
            
            # 2. Lấy giá trị PDD cho độ sâu này
            pdd_data = self.beam_models[energy]["PDD"]
            depths = pdd_data["depths"]
            pdd_values = pdd_data["values"]
            
            # Nội suy để lấy giá trị PDD tại độ sâu
            pdd = np.interp(depth, depths, pdd_values)
            
            # 3. Tính toán liều tại điểm
            # Công thức đơn giản: Dose = MU × OF × PDD × ISF
            # OF: Output Factor
            # ISF: Inverse Square Factor (tỷ lệ nghịch bình phương)
            
            # Lấy output factor cho kích thước trường
            fs_data = self.beam_models[energy]["output_factors"]
            field_sizes = fs_data["field_sizes"]
            of_values = fs_data["values"]
            
            # Lấy kích thước trường tương đương
            eq_field_size = np.sqrt(field_size_cm[0] * field_size_cm[1])
            of = np.interp(eq_field_size, field_sizes, of_values)
            
            # Tính ISF
            isf = (100.0 / sad) ** 2  # Giả sử normalization ở SSD = 100cm
            
            # Tính liều
            dose = mu * of * pdd * isf / 100.0  # Chia 100 vì PDD là phần trăm
            
            return dose
            
        except Exception as e:
            logger.error(f"Error calculating point dose: {e}")
            return 0.0
    
    def calculate_dose_distribution(self, beam_params: Dict, volume_dimensions: List[int], 
                                   voxel_size: List[float], algorithm: str = "simple") -> np.ndarray:
        """
        Tính toán phân bố liều trong một thể tích
        
        Args:
            beam_params: Thông số chùm tia
            volume_dimensions: Kích thước thể tích [nx, ny, nz]
            voxel_size: Kích thước voxel [dx, dy, dz] (mm)
            algorithm: Thuật toán tính liều ("simple", "collapsed_cone", "monte_carlo")
            
        Returns:
            np.ndarray: Mảng 3D chứa phân bố liều
        """
        try:
            # Tạo mảng liều
            dose_grid = np.zeros(volume_dimensions)
            
            # Đánh dấu đang tính toán
            self.calculation_in_progress = True
            
            # Lựa chọn thuật toán tính liều
            if algorithm == "simple":
                dose_grid = self._calculate_simple_dose(beam_params, volume_dimensions, voxel_size)
            elif algorithm == "collapsed_cone":
                logger.info("Collapsed cone algorithm not fully implemented, using simple dose calculation")
                dose_grid = self._calculate_simple_dose(beam_params, volume_dimensions, voxel_size)
            elif algorithm == "monte_carlo":
                logger.info("Monte Carlo algorithm not fully implemented, using simple dose calculation")
                dose_grid = self._calculate_simple_dose(beam_params, volume_dimensions, voxel_size)
            else:
                logger.error(f"Unknown dose calculation algorithm: {algorithm}")
                self.calculation_in_progress = False
                return dose_grid
            
            # Chuẩn hóa liều
            if np.max(dose_grid) > 0:
                dose_grid = dose_grid / np.max(dose_grid) * beam_params.get("prescription_dose", 2.0)  # Gy
            
            self.calculation_in_progress = False
            return dose_grid
            
        except Exception as e:
            logger.error(f"Error calculating dose distribution: {e}")
            self.calculation_in_progress = False
            return np.zeros(volume_dimensions)
    
    def _calculate_simple_dose(self, beam_params: Dict, volume_dimensions: List[int], 
                              voxel_size: List[float]) -> np.ndarray:
        """
        Tính phân bố liều đơn giản
        
        Args:
            beam_params: Thông số chùm tia
            volume_dimensions: Kích thước thể tích [nx, ny, nz]
            voxel_size: Kích thước voxel [dx, dy, dz] (mm)
            
        Returns:
            np.ndarray: Mảng 3D chứa phân bố liều
        """
        # Tạo mảng liều
        dose_grid = np.zeros(volume_dimensions)
        
        # Lấy thông số chùm tia
        energy = beam_params.get("energy", "6MV")
        isocenter = beam_params.get("isocenter", [0, 0, 0])  # mm
        field_size = beam_params.get("field_size", [100, 100])  # mm
        gantry_angle = np.radians(beam_params.get("gantry_angle", 0))  # chuyển từ độ sang radian
        collimator_angle = np.radians(beam_params.get("collimator_angle", 0))  # chuyển từ độ sang radian
        
        # Tạo tọa độ voxel
        nx, ny, nz = volume_dimensions
        dx, dy, dz = voxel_size
        
        x = np.arange(nx) * dx - (nx * dx) / 2 + isocenter[0]
        y = np.arange(ny) * dy - (ny * dy) / 2 + isocenter[1]
        z = np.arange(nz) * dz - (nz * dz) / 2 + isocenter[2]
        
        # Tính ma trận xoay cho góc gantry
        rotation_matrix = np.array([
            [np.cos(gantry_angle), 0, np.sin(gantry_angle)],
            [0, 1, 0],
            [-np.sin(gantry_angle), 0, np.cos(gantry_angle)]
        ])
        
        # Tính phân bố liều
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Vị trí voxel
                    point = np.array([x[i], y[j], z[k]])
                    
                    # Xoay điểm theo góc gantry
                    point_rotated = np.dot(rotation_matrix, point - np.array(isocenter)) + np.array(isocenter)
                    
                    # Tính khoảng cách đến trục trung tâm
                    dist_to_axis = np.sqrt(point_rotated[0] ** 2 + point_rotated[1] ** 2)
                    
                    # Tính độ sâu
                    depth = point_rotated[2]
                    
                    # Kiểm tra xem điểm có nằm trong trường xạ không
                    if (abs(point_rotated[0]) <= field_size[0] / 2 and 
                        abs(point_rotated[1]) <= field_size[1] / 2):
                        
                        # Lấy dữ liệu PDD
                        pdd_data = self.beam_models[energy]["PDD"]
                        depths = np.array(pdd_data["depths"]) * 10  # Convert to mm
                        pdd_values = pdd_data["values"]
                        
                        # Nội suy PDD
                        pdd = np.interp(depth, depths, pdd_values, left=0, right=0)
                        
                        # Mô phỏng off-axis ratio
                        sigma = 0.35 * np.mean(field_size) / 10  # cm
                        oar = np.exp(-(dist_to_axis/10) ** 2 / (2 * sigma ** 2))
                        
                        # Tính liều
                        dose_grid[i, j, k] = pdd * oar
                    else:
                        # Mô phỏng penumbra
                        min_dist = min(
                            abs(point_rotated[0]) - field_size[0] / 2,
                            abs(point_rotated[1]) - field_size[1] / 2
                        )
                        
                        if min_dist < 20:  # mm
                            # Penumbra region
                            sigma = 5.0  # mm
                            dose_grid[i, j, k] = np.exp(-min_dist ** 2 / (2 * sigma ** 2))
        
        return dose_grid
    
    def calculate_dvh(self, dose_grid: np.ndarray, structure_mask: np.ndarray, 
                     bins: int = 100, dose_max: Optional[float] = None) -> Dict:
        """
        Tính Dose-Volume Histogram (DVH)
        
        Args:
            dose_grid: Mảng 3D chứa phân bố liều
            structure_mask: Mảng nhị phân đánh dấu cấu trúc
            bins: Số bin trong histogram
            dose_max: Liều tối đa để tính (nếu None, sử dụng max(dose_grid))
            
        Returns:
            Dict: Dữ liệu DVH
        """
        try:
            # Kiểm tra kích thước
            if dose_grid.shape != structure_mask.shape:
                logger.error(f"Dose grid and structure mask have different shapes: {dose_grid.shape} vs {structure_mask.shape}")
                return {}
            
            # Lấy các voxel thuộc cấu trúc
            structure_voxels = dose_grid[structure_mask > 0]
            
            if len(structure_voxels) == 0:
                logger.warning("No voxels in structure mask")
                return {
                    "dose": [],
                    "volume_ratio": [],
                    "volume_cc": [],
                    "min_dose": 0,
                    "max_dose": 0,
                    "mean_dose": 0,
                    "median_dose": 0,
                    "volume_cc": 0
                }
            
            # Xác định liều tối đa
            if dose_max is None:
                dose_max = np.max(structure_voxels)
                if dose_max == 0:
                    dose_max = 1.0  # Để tránh chia cho 0
            
            # Tạo bins
            dose_bins = np.linspace(0, dose_max, bins + 1)
            
            # Tính histogram
            hist, bin_edges = np.histogram(structure_voxels, bins=dose_bins)
            
            # Tính DVH tích lũy
            dvh = np.zeros_like(hist, dtype=float)
            for i in range(len(hist)):
                dvh[i] = np.sum(hist[i:])
            
            # Chuẩn hóa thành phần trăm
            total_voxels = len(structure_voxels)
            dvh = (dvh / total_voxels) * 100
            
            # Thống kê
            min_dose = np.min(structure_voxels)
            max_dose = np.max(structure_voxels)
            mean_dose = np.mean(structure_voxels)
            median_dose = np.median(structure_voxels)
            
            # Ước tính thể tích (giả sử voxel size = 1mm^3)
            volume_cc = total_voxels / 1000.0  # cc
            
            return {
                "dose": bin_edges[:-1].tolist(),
                "volume_ratio": dvh.tolist(),
                "volume_cc": (dvh * volume_cc / 100.0).tolist(),
                "min_dose": float(min_dose),
                "max_dose": float(max_dose),
                "mean_dose": float(mean_dose),
                "median_dose": float(median_dose),
                "volume_cc": float(volume_cc)
            }
            
        except Exception as e:
            logger.error(f"Error calculating DVH: {e}")
            return {}
    
    def calculate_eud(self, dose_grid: np.ndarray, structure_mask: np.ndarray, a: float) -> float:
        """
        Tính Equivalent Uniform Dose (EUD)
        
        Args:
            dose_grid: Mảng 3D chứa phân bố liều
            structure_mask: Mảng nhị phân đánh dấu cấu trúc
            a: Thông số a trong công thức EUD
            
        Returns:
            float: Giá trị EUD
        """
        try:
            # Lấy các voxel thuộc cấu trúc
            structure_voxels = dose_grid[structure_mask > 0]
            
            if len(structure_voxels) == 0:
                logger.warning("No voxels in structure mask for EUD calculation")
                return 0.0
            
            # Tính EUD
            if abs(a) < 1e-6:
                # Trường hợp a gần 0, sử dụng log
                eud = np.exp(np.mean(np.log(structure_voxels + 1e-10)))
            else:
                # Công thức tổng quát
                eud = (np.mean(structure_voxels ** a)) ** (1 / a)
            
            return float(eud)
            
        except Exception as e:
            logger.error(f"Error calculating EUD: {e}")
            return 0.0
    
    def calculate_ntcp(self, dose_grid: np.ndarray, structure_mask: np.ndarray, 
                      td50: float, gamma_50: float, a: float) -> float:
        """
        Tính Normal Tissue Complication Probability (NTCP)
        
        Args:
            dose_grid: Mảng 3D chứa phân bố liều
            structure_mask: Mảng nhị phân đánh dấu cấu trúc
            td50: Liều gây tổn thương 50% (Gy)
            gamma_50: Độ dốc của đường cong NTCP
            a: Thông số a trong công thức EUD
            
        Returns:
            float: Giá trị NTCP
        """
        try:
            # Tính EUD
            eud = self.calculate_eud(dose_grid, structure_mask, a)
            
            # Tính NTCP theo mô hình LKB
            t = (eud - td50) / (td50 * gamma_50 / 100.0)
            ntcp = 1.0 / (1.0 + np.exp(-t))
            
            return float(ntcp)
            
        except Exception as e:
            logger.error(f"Error calculating NTCP: {e}")
            return 0.0
    
    def calculate_tcp(self, dose_grid: np.ndarray, structure_mask: np.ndarray, 
                    tcd50: float, gamma_50: float) -> float:
        """
        Tính Tumor Control Probability (TCP)
        
        Args:
            dose_grid: Mảng 3D chứa phân bố liều
            structure_mask: Mảng nhị phân đánh dấu cấu trúc
            tcd50: Liều kiểm soát khối u 50% (Gy)
            gamma_50: Độ dốc của đường cong TCP
            
        Returns:
            float: Giá trị TCP
        """
        try:
            # Lấy các voxel thuộc cấu trúc
            structure_voxels = dose_grid[structure_mask > 0]
            
            if len(structure_voxels) == 0:
                logger.warning("No voxels in structure mask for TCP calculation")
                return 0.0
            
            # Tính TCP cho từng voxel
            mean_dose = np.mean(structure_voxels)
            t = (mean_dose - tcd50) / (tcd50 * gamma_50 / 100.0)
            tcp = 1.0 / (1.0 + np.exp(-t))
            
            return float(tcp)
            
        except Exception as e:
            logger.error(f"Error calculating TCP: {e}")
            return 0.0

# Singleton instance
_instance = None

def get_instance() -> DoseCalculator:
    """
    Lấy instance của DoseCalculator (Singleton pattern)
    
    Returns:
        DoseCalculator: Instance của dose calculator
    """
    global _instance
    if _instance is None:
        _instance = DoseCalculator()
    return _instance

def initialize():
    """Khởi tạo hệ thống tính toán liều"""
    return get_instance().initialize()
