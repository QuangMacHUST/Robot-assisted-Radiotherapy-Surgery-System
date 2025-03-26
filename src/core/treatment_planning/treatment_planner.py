#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboRadAssist - Module lập kế hoạch điều trị
Cung cấp các chức năng lập kế hoạch xạ trị, tối ưu hóa góc chiếu và liều lượng
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import threading
from pathlib import Path
import time

# Import từ các module khác
from src.core.dose_calculation import dose_calculator
from src.core.image_processing import image_processor

logger = logging.getLogger(__name__)

class TreatmentPlanner:
    """Class quản lý và lập kế hoạch điều trị xạ trị"""
    
    def __init__(self):
        """Khởi tạo Treatment Planner"""
        self.current_plan = None
        self.available_plans = {}
        self.optimization_in_progress = False
        self.optimization_results = {}
        self.plan_evaluations = {}
        self.mutex = threading.Lock()
        
    def initialize(self):
        """Khởi tạo hệ thống lập kế hoạch điều trị"""
        logger.info("Initializing treatment planning system")
        try:
            # Thực hiện các bước khởi tạo cần thiết
            logger.info("Treatment planning system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize treatment planning system: {e}")
            return False
    
    def create_new_plan(self, patient_id: str, plan_name: str, 
                        prescription: Dict, structures: Dict, modality: str = "IMRT") -> str:
        """
        Tạo kế hoạch điều trị mới
        
        Args:
            patient_id: ID của bệnh nhân
            plan_name: Tên kế hoạch
            prescription: Dict chứa thông tin kê toa (liều, phân đoạn)
            structures: Dict chứa thông tin về các cấu trúc mục tiêu và cơ quan nguy cơ
            modality: Phương thức điều trị (IMRT, VMAT, SBRT, ...)
            
        Returns:
            str: ID của kế hoạch mới
        """
        try:
            plan_id = f"{patient_id}_{plan_name}_{int(time.time())}"
            
            new_plan = {
                'id': plan_id,
                'patient_id': patient_id,
                'name': plan_name,
                'prescription': prescription,
                'structures': structures,
                'modality': modality,
                'created_at': time.time(),
                'modified_at': time.time(),
                'status': 'created',
                'beams': [],
                'optimization_parameters': {},
                'dose_grid': None
            }
            
            with self.mutex:
                self.available_plans[plan_id] = new_plan
                self.current_plan = new_plan
                
            logger.info(f"Created new treatment plan: {plan_id}")
            return plan_id
            
        except Exception as e:
            logger.error(f"Error creating treatment plan: {e}")
            return ""
    
    def add_beam(self, plan_id: str, beam_params: Dict) -> bool:
        """
        Thêm chùm tia vào kế hoạch
        
        Args:
            plan_id: ID của kế hoạch
            beam_params: Thông số chùm tia
            
        Returns:
            bool: True nếu thêm thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        try:
            beam_id = len(self.available_plans[plan_id]['beams']) + 1
            
            beam = {
                'id': beam_id,
                'name': beam_params.get('name', f"Beam_{beam_id}"),
                'gantry_angle': beam_params.get('gantry_angle', 0),
                'collimator_angle': beam_params.get('collimator_angle', 0),
                'couch_angle': beam_params.get('couch_angle', 0),
                'energy': beam_params.get('energy', '6MV'),
                'isocenter': beam_params.get('isocenter', [0, 0, 0]),
                'mlc': beam_params.get('mlc', None),
                'weight': beam_params.get('weight', 1.0)
            }
            
            with self.mutex:
                self.available_plans[plan_id]['beams'].append(beam)
                self.available_plans[plan_id]['modified_at'] = time.time()
                
            logger.info(f"Added beam {beam_id} to plan {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding beam to plan: {e}")
            return False
    
    def set_optimization_parameters(self, plan_id: str, parameters: Dict) -> bool:
        """
        Đặt tham số tối ưu hóa cho kế hoạch
        
        Args:
            plan_id: ID của kế hoạch
            parameters: Các tham số tối ưu hóa
            
        Returns:
            bool: True nếu đặt thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        try:
            with self.mutex:
                self.available_plans[plan_id]['optimization_parameters'] = parameters
                self.available_plans[plan_id]['modified_at'] = time.time()
                
            logger.info(f"Set optimization parameters for plan {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting optimization parameters: {e}")
            return False
    
    def optimize_plan(self, plan_id: str, algorithm: str = "IPOPT", max_iterations: int = 100) -> bool:
        """
        Tối ưu hóa kế hoạch điều trị
        
        Args:
            plan_id: ID của kế hoạch
            algorithm: Thuật toán tối ưu hóa
            max_iterations: Số lần lặp tối đa
            
        Returns:
            bool: True nếu bắt đầu tối ưu hóa thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        if self.optimization_in_progress:
            logger.error("Optimization already in progress")
            return False
            
        try:
            # Kiểm tra xem kế hoạch có đủ thông tin cần thiết không
            plan = self.available_plans[plan_id]
            
            if not plan['beams']:
                logger.error("No beams defined in the plan")
                return False
                
            if not plan['optimization_parameters']:
                logger.error("No optimization parameters defined")
                return False
            
            # Đánh dấu đang tối ưu hóa
            self.optimization_in_progress = True
            with self.mutex:
                self.available_plans[plan_id]['status'] = 'optimizing'
            
            # Thực hiện tối ưu hóa trong thread riêng để không chặn luồng chính
            threading.Thread(
                target=self._run_optimization, 
                args=(plan_id, algorithm, max_iterations),
                daemon=True
            ).start()
            
            logger.info(f"Started optimization for plan {plan_id} using {algorithm}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting optimization: {e}")
            self.optimization_in_progress = False
            return False
    
    def _run_optimization(self, plan_id: str, algorithm: str, max_iterations: int):
        """
        Hàm thực hiện tối ưu hóa (chạy trong thread riêng)
        
        Args:
            plan_id: ID của kế hoạch
            algorithm: Thuật toán tối ưu hóa
            max_iterations: Số lần lặp tối đa
        """
        try:
            logger.info(f"Running optimization for plan {plan_id}")
            
            plan = self.available_plans[plan_id]
            
            # Mô phỏng quá trình tối ưu hóa
            iterations = min(max_iterations, 20)  # Mô phỏng, giới hạn số lần lặp
            
            for i in range(iterations):
                # Kiểm tra xem có bị dừng không
                if not self.optimization_in_progress:
                    logger.info("Optimization stopped")
                    break
                
                # Mô phỏng một bước tối ưu hóa
                time.sleep(0.5)  # Mô phỏng thời gian tính toán
                
                # Cập nhật trạng thái
                progress = (i + 1) / iterations
                logger.info(f"Optimization progress: {progress*100:.1f}%")
                
                # Trong phiên bản thực tế, đây sẽ là các bước:
                # 1. Tính toán phân bố liều từ trọng số chùm tia và MLC hiện tại
                # 2. Tính toán gradient hàm mục tiêu
                # 3. Cập nhật trọng số chùm tia và vị trí MLC
                # 4. Kiểm tra điều kiện hội tụ
            
            # Mô phỏng kết quả tối ưu hóa đơn giản
            optimization_result = {
                'algorithm': algorithm,
                'iterations_performed': iterations,
                'objective_value': 0.85,  # Mô phỏng
                'constraints_violated': 0,
                'completion_status': 'completed',
                'completion_time': time.time()
            }
            
            # Mô phỏng cập nhật kế hoạch (trong hệ thống thực, đây sẽ cập nhật trọng số chùm tia và MLC)
            with self.mutex:
                # Cập nhật trọng số chùm tia đơn giản
                num_beams = len(self.available_plans[plan_id]['beams'])
                if num_beams > 0:
                    for i in range(num_beams):
                        # Đặt trọng số giả định
                        self.available_plans[plan_id]['beams'][i]['weight'] = 1.0 / num_beams
                
                # Cập nhật trạng thái kế hoạch
                self.available_plans[plan_id]['status'] = 'optimized'
                self.available_plans[plan_id]['modified_at'] = time.time()
                
                # Lưu kết quả tối ưu hóa
                self.optimization_results[plan_id] = optimization_result
            
            logger.info(f"Optimization completed for plan {plan_id}")
            
            # Tính toán phân bố liều cho kế hoạch đã tối ưu hóa
            self.calculate_dose(plan_id)
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            with self.mutex:
                self.available_plans[plan_id]['status'] = 'error'
        finally:
            self.optimization_in_progress = False
    
    def calculate_dose(self, plan_id: str) -> bool:
        """
        Tính toán phân bố liều cho kế hoạch
        
        Args:
            plan_id: ID của kế hoạch
            
        Returns:
            bool: True nếu tính toán thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        try:
            logger.info(f"Calculating dose for plan {plan_id}")
            plan = self.available_plans[plan_id]
            
            # Trong phiên bản thực tế, đây sẽ gọi module dose_calculator
            # Ở đây, chúng ta tạo dữ liệu mô phỏng
            
            # Tạo lưới liều đơn giản
            grid_size = (50, 50, 30)  # (x, y, z)
            dose_grid = np.zeros(grid_size)
            
            # Mô phỏng phân bố liều đơn giản
            center = np.array(grid_size) // 2
            
            for beam in plan['beams']:
                # Mô phỏng đóng góp liều của mỗi chùm tia
                weight = beam['weight']
                gantry_angle = beam['gantry_angle']
                
                # Mô phỏng đơn giản: tạo một vùng liều cao theo hướng của chùm tia
                for z in range(grid_size[2]):
                    for y in range(grid_size[1]):
                        for x in range(grid_size[0]):
                            # Tính khoảng cách đến tâm
                            dx, dy, dz = x - center[0], y - center[1], z - center[2]
                            
                            # Mô phỏng đơn giản một chùm tia
                            angle_rad = np.radians(gantry_angle)
                            beam_direction = np.array([np.sin(angle_rad), np.cos(angle_rad), 0])
                            point = np.array([dx, dy, dz])
                            
                            # Tính projection của điểm lên hướng chùm tia
                            proj = np.dot(point, beam_direction) / np.linalg.norm(beam_direction)
                            
                            # Khoảng cách đến trục chùm tia
                            dist_to_axis = np.linalg.norm(point - proj * beam_direction)
                            
                            # Mô phỏng suy giảm liều theo khoảng cách
                            dose_contribution = weight * np.exp(-0.1 * dist_to_axis) * np.exp(-0.05 * abs(proj))
                            
                            # Thêm đóng góp vào lưới liều
                            dose_grid[x, y, z] += dose_contribution
            
            # Chuẩn hóa lưới liều so với liều kê
            prescription_dose = plan['prescription'].get('dose', 60.0)  # Gy
            max_dose = np.max(dose_grid)
            if max_dose > 0:
                dose_grid = dose_grid / max_dose * prescription_dose
            
            # Lưu lưới liều vào kế hoạch
            with self.mutex:
                self.available_plans[plan_id]['dose_grid'] = dose_grid
                self.available_plans[plan_id]['status'] = 'dose_calculated'
                
            logger.info(f"Dose calculation completed for plan {plan_id}")
            
            # Đánh giá kế hoạch
            self.evaluate_plan(plan_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating dose: {e}")
            return False
    
    def evaluate_plan(self, plan_id: str) -> Dict:
        """
        Đánh giá kế hoạch điều trị
        
        Args:
            plan_id: ID của kế hoạch
            
        Returns:
            Dict: Kết quả đánh giá kế hoạch
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return {}
            
        plan = self.available_plans[plan_id]
        
        if plan['dose_grid'] is None:
            logger.error(f"No dose grid available for plan {plan_id}")
            return {}
            
        try:
            logger.info(f"Evaluating plan {plan_id}")
            
            # Lấy lưới liều và thông tin kê toa
            dose_grid = plan['dose_grid']
            prescription = plan['prescription']
            prescription_dose = prescription.get('dose', 60.0)  # Gy
            structures = plan['structures']
            
            # Tạo các chỉ số đánh giá mô phỏng
            evaluation = {
                'plan_id': plan_id,
                'time': time.time(),
                'metrics': {
                    'PTV': {
                        'D95': prescription_dose * 0.95,  # Mô phỏng
                        'D50': prescription_dose,
                        'D5': prescription_dose * 1.05,
                        'V95': 98.5,  # % của thể tích nhận ít nhất 95% liều kê
                        'homogeneity_index': 0.07,  # (D5-D95)/D50
                        'conformity_index': 0.85  # Tỷ lệ thể tích chiếu xạ / thể tích PTV
                    }
                }
            }
            
            # Thêm chỉ số cho các cơ quan nguy cơ
            for organ in structures.get('OARs', []):
                # Mô phỏng liều tối đa/trung bình cho các cơ quan
                evaluation['metrics'][organ] = {
                    'Dmax': prescription_dose * 0.7,  # Mô phỏng
                    'Dmean': prescription_dose * 0.3
                }
            
            # Lưu kết quả đánh giá
            with self.mutex:
                self.plan_evaluations[plan_id] = evaluation
                
            logger.info(f"Plan evaluation completed for {plan_id}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating plan: {e}")
            return {}
    
    def export_plan(self, plan_id: str, output_path: Union[str, Path], format: str = 'DICOM') -> bool:
        """
        Xuất kế hoạch điều trị
        
        Args:
            plan_id: ID của kế hoạch
            output_path: Đường dẫn đến thư mục đầu ra
            format: Định dạng xuất (DICOM, PDF, ...)
            
        Returns:
            bool: True nếu xuất thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        try:
            output_path = Path(output_path)
            os.makedirs(output_path, exist_ok=True)
            
            logger.info(f"Exporting plan {plan_id} to {output_path} in {format} format")
            
            if format == 'DICOM':
                # Trong phiên bản thực tế, đây sẽ sử dụng thư viện pydicom để tạo file DICOM RTPlan
                # Ở đây, chúng ta mô phỏng việc xuất file
                
                plan_file_path = output_path / f"{plan_id}_RTPlan.dcm"
                dose_file_path = output_path / f"{plan_id}_RTDose.dcm"
                
                # Mô phỏng việc lưu file
                with open(plan_file_path, 'w') as f:
                    f.write("DICOM RTPlan - Mockup")
                
                with open(dose_file_path, 'w') as f:
                    f.write("DICOM RTDose - Mockup")
                
                logger.info(f"Exported DICOM RTPlan to {plan_file_path}")
                logger.info(f"Exported DICOM RTDose to {dose_file_path}")
                
            elif format == 'PDF':
                # Mô phỏng tạo báo cáo PDF
                report_file_path = output_path / f"{plan_id}_Report.pdf"
                
                # Mô phỏng việc lưu file
                with open(report_file_path, 'w') as f:
                    f.write("Treatment Plan Report - Mockup")
                
                logger.info(f"Exported PDF report to {report_file_path}")
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error exporting plan: {e}")
            return False
    
    def get_plan(self, plan_id: str) -> Dict:
        """
        Lấy thông tin kế hoạch
        
        Args:
            plan_id: ID của kế hoạch
            
        Returns:
            Dict: Thông tin kế hoạch
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return {}
        
        # Trả về bản sao của kế hoạch để tránh sửa đổi trực tiếp
        plan_copy = dict(self.available_plans[plan_id])
        
        # Không bao gồm dữ liệu liều lớn trong kết quả
        if 'dose_grid' in plan_copy:
            # Chỉ bao gồm kích thước lưới liều, không phải toàn bộ dữ liệu
            dose_grid = plan_copy['dose_grid']
            if dose_grid is not None:
                plan_copy['dose_grid_info'] = {
                    'shape': dose_grid.shape,
                    'min': float(np.min(dose_grid)),
                    'max': float(np.max(dose_grid)),
                    'mean': float(np.mean(dose_grid))
                }
            plan_copy.pop('dose_grid')
            
        return plan_copy
    
    def list_plans(self, patient_id: Optional[str] = None) -> List[Dict]:
        """
        Liệt kê các kế hoạch
        
        Args:
            patient_id: ID của bệnh nhân (nếu None, liệt kê tất cả kế hoạch)
            
        Returns:
            List[Dict]: Danh sách thông tin tóm tắt các kế hoạch
        """
        plans = []
        
        for plan_id, plan in self.available_plans.items():
            if patient_id is None or plan['patient_id'] == patient_id:
                # Chỉ bao gồm thông tin tóm tắt
                summary = {
                    'id': plan_id,
                    'name': plan['name'],
                    'patient_id': plan['patient_id'],
                    'modality': plan['modality'],
                    'status': plan['status'],
                    'created_at': plan['created_at'],
                    'modified_at': plan['modified_at'],
                    'beam_count': len(plan['beams'])
                }
                plans.append(summary)
                
        return plans
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Xóa kế hoạch
        
        Args:
            plan_id: ID của kế hoạch
            
        Returns:
            bool: True nếu xóa thành công
        """
        if plan_id not in self.available_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
            
        with self.mutex:
            # Nếu đây là kế hoạch hiện tại, đặt lại current_plan
            if self.current_plan and self.current_plan['id'] == plan_id:
                self.current_plan = None
                
            # Xóa kế hoạch
            del self.available_plans[plan_id]
            
            # Xóa các kết quả liên quan
            if plan_id in self.optimization_results:
                del self.optimization_results[plan_id]
                
            if plan_id in self.plan_evaluations:
                del self.plan_evaluations[plan_id]
                
        logger.info(f"Deleted plan {plan_id}")
        return True

# Singleton instance
_instance = None

def get_instance() -> TreatmentPlanner:
    """
    Lấy instance của TreatmentPlanner (Singleton pattern)
    
    Returns:
        TreatmentPlanner: Instance của treatment planner
    """
    global _instance
    if _instance is None:
        _instance = TreatmentPlanner()
    return _instance

def initialize():
    """Khởi tạo hệ thống lập kế hoạch điều trị"""
    return get_instance().initialize()
