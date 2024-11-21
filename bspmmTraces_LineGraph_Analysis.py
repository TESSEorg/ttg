#!/usr/bin/env python3

import sys
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def extract_data_from_json(json_data):
    entries = []

    for event in json_data["traceEvents"]:
        pid = event["pid"]
        function_name = event["name"].split('<')[0]
        duration_ms = event["dur"]

        entry_data = {
            "pid": pid,
            "function_name": function_name,
            "duration": duration_ms
        }

        entries.append(entry_data)

    return entries

def generate_curves(entries, target_pid):
    function_histograms = {}  # Create a dictionary to store durations for each function

    for entry in entries:
        pid = entry["pid"]
        function_name = entry["function_name"]
        duration_ms = entry["duration"] / 1000  # Convert ms to sec

        if pid == target_pid:
            if function_name not in function_histograms:
                function_histograms[function_name] = []

            function_histograms[function_name].append(duration_ms)

    pdf_filename = f'curves_for_pid_{target_pid}.pdf'
    pdf_pages = PdfPages(pdf_filename)

    for function_name, durations in function_histograms.items():
        plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
        plt.plot(durations, marker='o', linestyle='-', markersize=5)  # Create a line plot
        plt.xlabel('Entry')
        plt.ylabel('Duration (s)')
        plt.title(f'Curve of Durations for Function {function_name} (PID {target_pid})')

        # Add annotations for each data point
        for i, duration in enumerate(durations):
            plt.annotate(f'{duration:.2f}s', (i, duration), textcoords="offset points", xytext=(0,10), ha='center')

        pdf_pages.savefig()
        plt.close()

        # Compute and print average and standard deviation
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        print(f"Function: {function_name} (PID {target_pid})")
        print(f"Average Duration: {avg_duration:.2f} s")
        print(f"Standard Deviation: {std_duration:.2f} s")
        print("")

    pdf_pages.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python_script.py function_data.json pid_value")
        return
    
    json_file_path = sys.argv[1]
    
    try:
        target_pid = int(sys.argv[2])
    except ValueError:
        print("Error: Invalid pid_value. Please enter a valid integer.")
        return

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    entries = extract_data_from_json(json_data)
    
    # Check if the target_pid exists in the data
    pids = set(entry["pid"] for entry in entries)
    if target_pid not in pids:
        print(f"Error: PID {target_pid} not found in the data.")
        return
    
    generate_curves(entries, target_pid)

if __name__ == "__main__":
    main()










