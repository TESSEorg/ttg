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

def generate_histograms(entries, target_pid):
    function_histograms = {}  # Create a dictionary to store durations for each function

    for entry in entries:
        pid = entry["pid"]
        function_name = entry["function_name"]
        duration_ms = entry["duration"] / 1000  # Convert ms to sec

        if pid == target_pid:
            if function_name not in function_histograms:
                function_histograms[function_name] = []

            function_histograms[function_name].append(duration_ms)

    pdf_filename = f'histograms_for_pid_{target_pid}.pdf'
    pdf_pages = PdfPages(pdf_filename)

    for function_name, durations in function_histograms.items():
        plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
        n, bins, patches = plt.hist(durations, bins=10, range=(0, max(durations)))  # Create a histogram
        plt.xlabel('Duration (s)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Durations for Function {function_name} (PID {target_pid})')

        # Add labels for each bar in the histogram
        for i, patch in enumerate(patches):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            label = f'{durations[i]:.2f}s'  # Label with exact duration value
            plt.annotate(label, (x, y), xytext=(0, 5), textcoords='offset points', ha='center')

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
    
    generate_histograms(entries, target_pid)

if __name__ == "__main__":
    main()

