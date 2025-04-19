import os
import time

def cleanup(report_path="concentration.out", loss_report_path="water_loss.out", water_usage_report_path="water_usage.out" ):
    if os.path.exists(report_path):
        for _ in range(10):
            try:
                os.remove(report_path)
                break
            except PermissionError:
                time.sleep(0.5)

    if os.path.exists(water_usage_report_path):
        for _ in range(10):
            try:
                os.remove(water_usage_report_path)
                break
            except PermissionError:
                time.sleep(0.5)

    if os.path.exists(loss_report_path):
        for _ in range(10):
            try:
                os.remove(loss_report_path)
                break
            except PermissionError:
                time.sleep(0.5)