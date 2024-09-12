#!/usr/bin/env python3

import sys
import json
import matplotlib.pyplot as plt
import statistics  # Import the statistics module for mean and other calculations

def generate_histogram(entry):
    functions_dur_counts = {}
    
    for event in entry:
        pid = event["pid"]
        function_name = event["name"].split('<')[0]
        duration = event["dur"]
        if pid not in functions_dur_counts:
            functions_dur_counts[pid] = {}
        functions_dur_counts[pid][function_name] = functions_dur_counts[pid].get(function_name, 0) + duration
    
    return functions_dur_counts

def generate_cumulative_histogram(entry):
    functions_cumulative_dur = {}

    for event in entry:
        pid = event["pid"]
        function_name = event["name"].split('::')[-1].split('<')[0]
        duration = event["dur"]
        if pid not in functions_cumulative_dur:
            functions_cumulative_dur[pid] = {}
        functions_cumulative_dur[pid][function_name] = functions_cumulative_dur[pid].get(function_name, 0) + duration

    return functions_cumulative_dur

def main():
    if len(sys.argv) != 2:
        print("Usage: python_script.py function_data.json")
        return
    
    json_file_path = sys.argv[1]

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    functions_dur_counts = generate_histogram(json_data["traceEvents"])

    # Histogram for individual PIDs
    for pid, function_dur_count in functions_dur_counts.items():
        total_duration = sum(duration for duration in function_dur_count.values()) / 1000  # Calculate total duration in seconds

        plt.bar(function_dur_count.keys(), [dur / 1000 for dur in function_dur_count.values()])  # Convert ms to sec
        plt.xlabel('Function Name')
        plt.ylabel('Total Duration (s)')  # Adding unit for time here
        plt.title(f'Histogram of Total Function Durations for PID {pid}')
        plt.xticks(rotation=45, ha='right')

        # Add total duration label to the top of each bar with rotation
        for function, duration in function_dur_count.items():
            plt.text(function, duration / 1000, f'{duration / 1000:.2f}', rotation=45, ha='center', va='bottom', fontweight='bold')  # Convert ms to sec

        # Add total duration to the top of the plot
        plt.text(0.5, 0.95, f'Total Duration: {total_duration:.2f} s', transform=plt.gca().transAxes, ha='center', fontweight='bold')

        plt.tight_layout()

        # Save each histogram as a separate PDF file
        pdf_filename = f'histogram_pid_{pid}.pdf'
        plt.savefig(pdf_filename)
        plt.close()

    # Histogram for cumulative time taken by PIDs for each function
    cumulative_function_dur_count = {}
    for function_dur_count in functions_dur_counts.values():
        for function, duration in function_dur_count.items():
            cumulative_function_dur_count[function] = cumulative_function_dur_count.get(function, 0) + duration

    plt.bar(cumulative_function_dur_count.keys(), [dur / 1000 for dur in cumulative_function_dur_count.values()])
    plt.xlabel('Function Name')
    plt.ylabel('Total Duration (s)')
    plt.title('Histogram of Cumulative Function Durations across PIDs')
    plt.xticks(rotation=45, ha='right')

    # Add total duration label to the top of each bar with rotation
    for function, duration in cumulative_function_dur_count.items():
        plt.text(function, duration / 1000, f'{duration / 1000:.2f}', rotation=45, ha='center', va='bottom', fontweight='bold')  # Convert ms to sec

    plt.tight_layout()

    # Save the cumulative histogram as a PDF file
    pdf_filename = 'cumulative_histogram.pdf'
    plt.savefig(pdf_filename)
    plt.close()

if __name__ == "__main__":
    main()
