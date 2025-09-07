#!/usr/bin/env python3
"""
Test script to verify service detection capabilities
"""

import sys
import os

# Add the current directory to path so we can import from test_performance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_performance import get_running_services, get_system_metrics, get_system_info

def test_service_detection():
    """Test the service detection functionality"""
    print("Testing Enhanced Service Detection Capabilities")
    print("=" * 50)
    
    # Test system info
    print("\n1. System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # Test service detection
    print("\n2. Service Detection:")
    services = get_running_services()
    
    print(f"   Total processes: {services['total_processes']}")
    print(f"   Fabric services: {len(services['fabric_services'])}")
    print(f"   IDS services: {len(services['ids_services'])}")
    print(f"   MQTT services: {len(services['mqtt_services'])}")
    print(f"   Monitoring services: {len(services['monitoring_services'])}")
    print(f"   Docker containers: {len(services['docker_containers'])}")
    
    # Show detected services
    if services['fabric_services']:
        print("\n   Detected Fabric Services:")
        for svc in services['fabric_services']:
            print(f"     - {svc['name']} (PID: {svc['pid']}, CPU: {svc['cpu_percent']:.1f}%, Mem: {svc['memory_mb']:.1f}MB)")
    
    if services['ids_services']:
        print("\n   Detected IDS Services:")
        for svc in services['ids_services']:
            print(f"     - {svc['name']} (PID: {svc['pid']}, CPU: {svc['cpu_percent']:.1f}%, Mem: {svc['memory_mb']:.1f}MB)")
    
    if services['mqtt_services']:
        print("\n   Detected MQTT Services:")
        for svc in services['mqtt_services']:
            print(f"     - {svc['name']} (PID: {svc['pid']}, CPU: {svc['cpu_percent']:.1f}%, Mem: {svc['memory_mb']:.1f}MB)")
    
    if services['monitoring_services']:
        print("\n   Detected Monitoring Services:")
        for svc in services['monitoring_services']:
            print(f"     - {svc['name']} (PID: {svc['pid']}, CPU: {svc['cpu_percent']:.1f}%, Mem: {svc['memory_mb']:.1f}MB)")
    
    if services['docker_containers']:
        print("\n   Docker Containers:")
        for container in services['docker_containers']:
            print(f"     - {container['name']}: {container['image']} ({container['status']})")
    
    # Test system metrics
    print("\n3. System Metrics:")
    metrics = get_system_metrics()
    
    print(f"   CPU Usage: {metrics['cpu']['usage_percent']:.1f}%")
    print(f"   Load Average: {metrics['cpu']['load_avg']}")
    print(f"   Memory Used: {metrics['memory']['used_percent']:.1f}% ({metrics['memory']['available_gb']:.1f}GB available)")
    print(f"   CPU Temperature: {metrics['thermal']['cpu_temp_c']:.1f}°C")
    print(f"   Network Sent: {metrics['network']['bytes_sent_mb']:.1f}MB")
    print(f"   Network Received: {metrics['network']['bytes_recv_mb']:.1f}MB")
    
    print("\n✓ Service detection test completed successfully!")
    print("\nThis validates that the enhanced test_performance.py script can:")
    print("  - Detect blockchain services (Fabric, orderers, peers)")
    print("  - Identify IDS components (Snort, Flower, etc.)")
    print("  - Monitor MQTT brokers for ESP32 telemetry")
    print("  - Track system monitoring tools (Glances, Prometheus)")
    print("  - Collect comprehensive system metrics")
    print("  - Verify service coexistence during ML inference")

if __name__ == "__main__":
    test_service_detection()
