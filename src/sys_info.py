import psutil
import platform
import cpuinfo
import gpuinfo
import os
import json

# Function to collect CPU information
def get_cpu_info():
    cpu_info = {}
    cpu_info['cpu_model'] = cpuinfo.get_cpu_info()['model']
    cpu_info['cpu_architecture'] = platform.machine()
    cpu_info['cpu_cores'] = psutil.cpu_count(logical=False)  # Physical cores
    cpu_info['cpu_threads'] = psutil.cpu_count(logical=True)  # Logical cores
    cpu_info['cpu_frequency'] = psutil.cpu_freq().current  # Current CPU frequency in MHz
    cpu_info['cpu_percent'] = psutil.cpu_percent(interval=1)  # Current CPU usage percentage
    return cpu_info

# Function to collect memory (RAM) information
def get_memory_info():
    memory_info = {}
    memory = psutil.virtual_memory()
    memory_info['total_memory_gb'] = memory.total / (1024 ** 3)  # in GB
    memory_info['available_memory_gb'] = memory.available / (1024 ** 3)  # in GB
    memory_info['used_memory_gb'] = memory.used / (1024 ** 3)  # in GB
    memory_info['memory_percent'] = memory.percent  # Memory usage percentage
    return memory_info

# Function to collect RAM details (brand, type, speed, etc.)
def get_ram_info():
    ram_info = {}
    if platform.system() == "Linux":
        try:
            import subprocess
            cmd = "dmidecode -t memory"
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result_output = result.stdout.decode()
            ram_info['ram_details'] = result_output
        except Exception as e:
            ram_info['ram_details'] = str(e)
    else:
        ram_info['ram_details'] = "Detailed RAM information is OS-specific and may not be available on Windows or MacOS."
    return ram_info

# Function to collect GPU information
def get_gpu_info():
    gpu_info = {}
    try:
        gpus = gpuinfo.get_info()
        gpu_info_list = []
        for gpu in gpus:
            info = {
                'gpu_model': gpu.name,
                'gpu_memory_total': gpu.memoryTotal / 1024,  # in GB
                'gpu_memory_free': gpu.memoryFree / 1024,  # in GB
                'gpu_memory_used': gpu.memoryUsed / 1024,  # in GB
                'gpu_utilization': gpu.utilization
            }
            gpu_info_list.append(info)
        gpu_info['gpus'] = gpu_info_list
    except Exception as e:
        gpu_info['gpus'] = str(e)
    return gpu_info

# Function to collect system disk information
def get_disk_info():
    disk_info = {}
    disk_info['disk_usage'] = psutil.disk_usage('/')
    disk_info['disk_free'] = disk_info['disk_usage'].free / (1024 ** 3)  # in GB
    disk_info['disk_total'] = disk_info['disk_usage'].total / (1024 ** 3)  # in GB
    disk_info['disk_used'] = disk_info['disk_usage'].used / (1024 ** 3)  # in GB
    disk_info['disk_percent'] = disk_info['disk_usage'].percent  # Disk usage percentage
    return disk_info

# Function to collect system information (CPU, GPU, Memory, Disk, etc.)
def collect_system_info():
    system_info = {}
    
    # Collect CPU information
    system_info['cpu_info'] = get_cpu_info()
    
    # Collect Memory information
    system_info['memory_info'] = get_memory_info()
    
    # Collect RAM specific details
    system_info['ram_info'] = get_ram_info()
    
    # Collect GPU information
    system_info['gpu_info'] = get_gpu_info()
    
    # Collect Disk information
    system_info['disk_info'] = get_disk_info()

    return system_info

# Main entry point to collect and print system information
if __name__ == "__main__":
    system_info = collect_system_info()
    
    # Print the information
    print(json.dumps(system_info, indent=4))
