#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

try:
    import os
    import numpy as np
    import time
    import pandas
    import sys
    import statistics
except ModuleNotFoundError:
    print("Did not find a system module, use pip to install it")

try:
    import parsec_trace_tables as ptt
    import pbt2ptt
except ModuleNotFoundError:
    print("Did not find pbt2ptt, you are likely using python version that does not match the version used to build PaRSEC profiling tools")
    print(sys.path)

import json
import re
import sys
import math

def bool(str):
    return str.lower() in ["true", "yes", "y", "1", "t"]

def validate_event_assignments(ctf_data):
    pid_to_function_map = {}

    # Iterate over the trace events and group function names by PID
    for event in ctf_data["traceEvents"]:
        pid = event["pid"]
        name = event["name"]
        
        if pid not in pid_to_function_map:
            pid_to_function_map[pid] = []
        
        pid_to_function_map[pid].append(name)
    
    # Print the grouped function names for each PID and event count
    for pid, functions in pid_to_function_map.items():
        print(f"PID: {pid}")
        for idx, function_name in enumerate(functions, start=1):
            print(f"{idx}. {function_name}")
        event_count = sum(1 for event in ctf_data["traceEvents"] if event.get("pid") == pid)
        print(f"Number of events with PID {pid}: {event_count}\n")

       
        
def pbt_to_ctf(pbt_files_list, ctf_filename, skip_parsec_events, skip_mpi_events, skip_fn_events):
    ctf_data = {"traceEvents": []}
    # Dictionary to store aggregated durations
    aggregated_durations = {}

    # Initialize lists to store duration values for each name
    duration_values = []
    ptt_filename = pbt2ptt.convert(pbt_files_list, multiprocess=False)
    trace = ptt.from_hdf(ptt_filename)

    # Variable to store the end time of MultiplyAdd operation
    multiplyadd_end_time = 0.0
    maximum_val = 0.0  # Initialize the maximum value variable

    for e in range(len(trace.events)):
        if skip_parsec_events == True and trace.event_names[trace.events.type[e]].startswith("PARSEC"):
            continue
        if skip_mpi_events == True and trace.event_names[trace.events.type[e]].startswith("MPI"):
            continue
        if skip_fn_events == True and trace.event_names[trace.events.type[e]].startswith("SpMM25D::MultiplyAdd"):
            print("MultiplyAdd Skipped successfully")
            continue

        ctf_event = {}
        ctf_event["ph"] = "X"  # complete event type
        ctf_event["ts"] = 0.001 * trace.events.begin[e]  # when we started, in ms
        ctf_event["dur"] = 0.001 * (trace.events.end[e] - trace.events.begin[e])  # when we started, in ms
        ctf_event["name"] = trace.event_names[trace.events.type[e]]

        if trace.events.key[e] is not None:
            ctf_event["args"] = trace.events.key[e].decode('utf-8').rstrip('\x00')
            ctf_event["name"] = trace.event_names[trace.events.type[e]] + "<" + ctf_event["args"] + ">"

        ctf_event["pid"] = trace.events.node_id[e]
        tid = trace.streams.th_id[trace.events.stream_id[e]]
        ctf_event["tid"] = 111111 if math.isnan(tid) else int(tid)

        ctf_data["traceEvents"].append(ctf_event)

        # Check if the current event is a MultiplyAdd event
        if trace.event_names[trace.events.type[e]].startswith("SpMM25D::MultiplyAdd"):
            current_val = round((ctf_event["ts"] + ctf_event["dur"]), 3)
            if current_val > maximum_val:
                maximum_val = current_val
                maximum_name = ctf_event["name"]  # Update the name of the function with maximum value
            print(f"", ctf_event["name"], " ends on ", current_val, "ms and pid =", ctf_event["pid"], "tid =", ctf_event["tid"])
            print("above event started on", round(ctf_event["ts"], 3), "and took duration of", round(ctf_event["dur"], 3))
            # Get the end time of the current MultiplyAdd operation
            multiplyadd_end_time = 0.001 * trace.events.end[e]
       
        
        # Get the index of the first occurrence of '<'
        index_of_open_bracket = ctf_event["name"].find('<')
        # Extract the substring before '<' and assign it to the name variable
        if index_of_open_bracket != -1:
            name = ctf_event["name"][:index_of_open_bracket]
            duration = ctf_event["dur"]
            if name in aggregated_durations:
                aggregated_durations[name]["duration"] += duration
                aggregated_durations[name]["count"] += 1
            else:
                # If name doesn't exist, create a new entry
                aggregated_durations[name] = {"duration": duration, "count": 1}
            # Add duration value to the list
            duration_values.append(duration)
    print("\n") 

    # Calculate the mean, median, max, min, and standard deviation for each aggregated duration
    for name, data in aggregated_durations.items():
        mean_duration = data["duration"] / data["count"]
        individual_durations = [ctf_event["dur"] for ctf_event in ctf_data["traceEvents"] if ctf_event["name"].startswith(name)]
        median_duration = statistics.median(individual_durations) if len(individual_durations) > 1 else 0.0
        max_duration = max(individual_durations)
        min_duration = min(individual_durations)
        std_deviation = statistics.stdev(individual_durations) if len(individual_durations) > 1 else 0.0
        print(f"Name: {name}, Mean: {mean_duration:.2f} μs, Median: {median_duration:.2f} μs, Max: {max_duration:.2f} μs, Min: {min_duration:.2f} μs, Std Deviation: {std_deviation:.2f} μs")
    print("\n")
    
    print("End time of MultiplyAdd operation:", multiplyadd_end_time, "ms")
    print("Last MultiplyAdd operation:", ctf_event["name"])

    # Print the name of the function with the maximum value
    print("Last MultiplyAdd Function with Maximum Duration Value:", maximum_name, " taking - ", maximum_val, "ms \n")

    # Run the validation function
    validate_event_assignments(ctf_data)

    with open(ctf_filename, "w") as chrome_trace:
        json.dump(ctf_data, chrome_trace)

if __name__ == "__main__":
    pbt_file_prefix = sys.argv[1]
    ctf_file_name = sys.argv[2]
    skip_parsec_events = True
    skip_mpi_events = True
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        sys.exit("usage: pbt_to_ctf.py <pbt base filename> <ctf filename> [skip PaRSEC events? default=1] [skip MPI events? default=1] [skip Function events? default=1]")
    if len(sys.argv) >= 4:
        skip_parsec_events = bool(sys.argv[3])
    if len(sys.argv) >= 5:
        skip_mpi_events = bool(sys.argv[4])
    if len(sys.argv) >= 6:
        skip_fn_events = bool(sys.argv[5])

    # iterate over all files within the directory that start with sys.argv[1]
    pbt_files_list = []
    dirname = os.path.dirname(pbt_file_prefix)
    for file in os.listdir(dirname):
        file_fullname = os.path.join(dirname, file)
        if file_fullname.startswith(pbt_file_prefix) and ".prof" in file_fullname and file_fullname != ctf_file_name:
            print("found file ", file_fullname)
            pbt_files_list.append(file_fullname)

    # to debug: read_pbt(pbt_files_list[0]), etc.
    pbt_to_ctf(pbt_files_list, ctf_file_name, skip_parsec_events, skip_mpi_events, skip_fn_events)
