#!/usr/bin/env python3

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

def read_pbt(pbt_files_list):
    ptt_filename = pbt2ptt.convert([pbt_files_list], multiprocess=False)
    trace = ptt.from_hdf(ptt_filename)

    print('The columns of the DataFrame (or data labels) and their datatypes are:')
    print(trace.events.dtypes)

    print('the types are:\n', trace.event_types)
    print('the streams are:\n', trace.streams)

    print('There are ' + str(len(trace.events)) + ' events in this trace', end=' ')
    for e in range(len(trace.events)):
        print('id===', trace.events.id[e], ' node_id=', trace.events.node_id[e], ' stream_id=', trace.events.stream_id[e], 'key=' ,trace.events.key[e], ' type=', trace.events.type[e], ' b=', trace.events.begin[e], ' e=', trace.events.end[e])

import json
import re
import sys
import math

def bool(str):
    return str.lower() in ["true", "yes", "y", "1", "t"]

def pbt_to_ctf(pbt_files_list, ctf_filename, skip_parsec_events, skip_mpi_events):
    ctf_data = {"traceEvents": []}
    # Dictionary to store aggregated durations
    aggregated_durations = {}

    # Initialize lists to store duration values for each name
    duration_values = []
    ptt_filename = pbt2ptt.convert(pbt_files_list, multiprocess=False)
    trace = ptt.from_hdf(ptt_filename)

    for e in range(len(trace.events)):
        # print('id=', trace.events.id[e], ' node_id=', trace.events.node_id[e], ' stream_id=', trace.events.stream_id[e], 'key=', trace.events.key[e], ' type=', trace.events.type[e], ' b=', trace.events.begin[e], ' e=', trace.events.end[e])
        # print('\n')

        if skip_parsec_events == True and trace.event_names[trace.events.type[e]].startswith("PARSEC"):
            continue
        if skip_mpi_events == True and trace.event_names[trace.events.type[e]].startswith("MPI"):
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

    # Calculate the mean, median, max, min, and standard deviation for each aggregated duration
    for name, data in aggregated_durations.items():
        mean_duration = data["duration"] / data["count"]
        individual_durations = [ctf_event["dur"] for ctf_event in ctf_data["traceEvents"] if ctf_event["name"].startswith(name)]
        median_duration = statistics.median(individual_durations) if len(individual_durations) > 1 else 0.0
        max_duration = max(individual_durations)
        min_duration = min(individual_durations)
        std_deviation = statistics.stdev(individual_durations) if len(individual_durations) > 1 else 0.0
        print(f"Name: {name}, Mean: {mean_duration:.2f} μs, Median: {median_duration:.2f} μs, Max: {max_duration:.2f} μs, Min: {min_duration:.2f} μs, Std Deviation: {std_deviation:.2f} μs")
    
    with open(ctf_filename, "w") as chrome_trace:
        json.dump(ctf_data, chrome_trace)

if __name__ == "__main__":
    pbt_file_prefix = sys.argv[1]
    ctf_file_name = sys.argv[2]
    skip_parsec_events = True
    skip_mpi_events = True
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        sys.exit("usage: pbt_to_ctf.py <pbt base filename> <ctf filename> [skip PaRSEC events? default=1] [skip MPI events? default=1]")
    if len(sys.argv) >= 4:
        skip_parsec_events = bool(sys.argv[3])
    if len(sys.argv) >= 5:
        skip_mpi_events = bool(sys.argv[4])

    # iterate over all files within the directory that start with sys.argv[1]
    pbt_files_list = []
    dirname = os.path.dirname(pbt_file_prefix)
    for file in os.listdir(dirname):
        file_fullname = os.path.join(dirname, file)
        if file_fullname.startswith(pbt_file_prefix) and ".prof" in file_fullname and file_fullname != ctf_file_name:
            print("found file ", file_fullname)
            pbt_files_list.append(file_fullname)

    # to debug: read_pbt(pbt_files_list[0]), etc.
    pbt_to_ctf(pbt_files_list, ctf_file_name, skip_parsec_events, skip_mpi_events)
