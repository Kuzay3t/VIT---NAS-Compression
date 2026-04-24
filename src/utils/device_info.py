# src/utils/device_info.py
import torch
import psutil
import platform
from typing import Dict, Any

class DeviceInfo:
    """Utility to detect and report system hardware."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_ghz': psutil.cpu_freq().current / 1000,
            'ram_gb': psutil.virtual_memory().total / (1024 ** 3),
            'available_ram_gb': psutil.virtual_memory().available / (1024 ** 3),
        }
    
    @staticmethod
    def get_torch_info() -> Dict[str, Any]:
        """Get PyTorch configuration."""
        return {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] 
                           if torch.cuda.is_available() else [],
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'cudnn_enabled': torch.backends.cudnn.enabled,
        }
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get detailed GPU information."""
        try:
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_info.append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2),
                    })
                return {'gpus': gpu_info}
            else:
                return {'error': 'No GPU available'}
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def print_system_info():
        """Print all system information."""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        
        sys_info = DeviceInfo.get_system_info()
        print("\n--- CPU & Memory ---")
        for key, value in sys_info.items():
            if isinstance(value, float):
                print(f"{key:.<30} {value:.2f}")
            else:
                print(f"{key:.<30} {value}")
        
        torch_info = DeviceInfo.get_torch_info()
        print("\n--- PyTorch ---")
        for key, value in torch_info.items():
            print(f"{key:.<30} {value}")
        
        gpu_info = DeviceInfo.get_gpu_info()
        print("\n--- GPU ---")
        if 'error' not in gpu_info:
            for gpu in gpu_info['gpus']:
                print(f"GPU {gpu['id']}: {gpu['name']}")
                print(f"  Memory: {gpu['memory_total_mb']:.0f} MB")
        else:
            print(f"Error: {gpu_info['error']}")
        
        print("="*60 + "\n")
