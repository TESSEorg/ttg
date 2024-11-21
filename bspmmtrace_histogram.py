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
        duration_us = event["dur"]
        timestamp_us = event["ts"]  # Added timestamp extraction

        entry_data = {
            "pid": pid,
            "function_name": function_name,
            "duration": duration_us,
            "timestamp": timestamp_us  # Added timestamp to entry data
        }

        entries.append(entry_data)

    return entries

def calculate_max_durations(entries):
    max_durations = {}

    for entry in entries:
        function_name = entry["function_name"]
        duration_us = entry["duration"]

        if function_name not in max_durations:
            max_durations[function_name] = duration_us
        else:
            max_durations[function_name] = max(max_durations[function_name], duration_us)

    return max_durations

def generate_histograms(entries, target_pid, max_durations, y_axis_buffer=1.1):
    function_histograms = {}
    last_event_durations = {}  # Store the duration of the last event for each function type

    for entry in entries:
        pid = entry["pid"]
        function_name = entry["function_name"]
        duration_us = entry["duration"]
        timestamp_us = entry["timestamp"]  # Extracted timestamp

        if pid == target_pid:
            if function_name not in function_histograms:
                function_histograms[function_name] = {"durations": [], "timestamps": [], "common_y_limit": None}

            function_histograms[function_name]["durations"].append(duration_us)
            function_histograms[function_name]["timestamps"].append(timestamp_us)

        # Update the duration of the last event for each function type
        if function_name not in last_event_durations:
            last_event_durations[function_name] = duration_us
        else:
            last_event_durations[function_name] = max(last_event_durations[function_name], duration_us)

    pdf_filename = f'histograms_pid_{target_pid}.pdf'
    pdf_pages = PdfPages(pdf_filename)

    for function_name, data in function_histograms.items():
        durations = data["durations"]
        timestamps = data["timestamps"]
        if not durations:
            print(f"No events found for PID {target_pid} and function {function_name}.")
            continue

        max_duration = max_durations[function_name]
        last_event_duration = last_event_durations[function_name]

        if data["common_y_limit"] is None:
            data["common_y_limit"] = len(durations)

        # Adjust the common y-axis limit by adding a buffer or scaling factor
        common_y_limit = int(data["common_y_limit"] * y_axis_buffer)

        plt.figure(figsize=(8, 6))
        bin_edges = np.linspace(0, max_duration, num=21)  # Adjusted x-axis limit
        n, bins, patches = plt.hist(durations, bins=bin_edges, range=(0, max_duration + 1))
        plt.xlabel('Duration (us)')
        plt.ylabel('Number of Events')
        plt.title(f'Histogram of Number of Events vs. Duration for Function {function_name} (PID {target_pid})')

        # Display the duration of the last event
        plt.text(0.7, 0.9, f'Last Event Duration: {last_event_duration} us', transform=plt.gca().transAxes)

        for bin_value, patch in zip(n, patches):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            label = f'{int(bin_value)}'
            plt.annotate(label, (x, y), xytext=(0, 5), textcoords='offset points', ha='center')

        # Set common y-axis limits with the adjusted buffer
        plt.xlim(0, max_duration)
        plt.ylim(0, common_y_limit)

        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        total_count = len(durations)

        plt.text(0.7, 0.8, f'Avg: {avg_duration:.2f} us', transform=plt.gca().transAxes)
        plt.text(0.7, 0.7, f'Std: {std_duration:.2f} us', transform=plt.gca().transAxes)

        pdf_pages.savefig()
        plt.close()

    pdf_pages.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python_script.py function_data.json")
        return
    
    json_file_path = sys.argv[1]

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    entries = extract_data_from_json(json_data)
    max_durations = calculate_max_durations(entries)

    for target_pid in set(entry["pid"] for entry in entries):
        generate_histograms(entries, target_pid, max_durations)

if __name__ == "__main__":
    main()

