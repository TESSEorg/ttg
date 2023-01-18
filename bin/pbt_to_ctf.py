#!/usr/bin/env python3

try:
    import os
    import numpy as np
    import time
    import pandas
    import sys
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
        print('id===', trace.events.id[e], ' node_id=', trace.events.node_id[e],' stream_id=',trace.events.stream_id[e], 'key=' ,trace.events.key[e],' type=',trace.events.type[e],' b=',trace.events.begin[e],' e=',trace.events.end[e])

import json
import re
import sys

def pbt_to_ctf(pbt_files_list, ctf_filename):

    ctf_data = {"traceEvents": []}
    proc = 0;
    print("list of all pbt files = ", [pbt_files_list])
    for filename in pbt_files_list:
        ptt_filename = pbt2ptt.convert([filename], multiprocess=False)
        trace = ptt.from_hdf(ptt_filename)
        # print(ptt_filename)
        # print(trace)
        # print(trace.events)

        index_of_proc= filename.find('.prof') - 1
        print("Proc", filename[index_of_proc], "is executing ", filename)
        for e in range(len(trace.events)):
            # print('id=',trace.events.id[e],' node_id=',trace.events.node_id[e],' stream_id=',trace.events.stream_id[e],'key=',trace.events.key[e],' type=',trace.events.type[e],' b=',trace.events.begin[e],' e=',trace.events.end[e])
            # print('\n')
            ctf_event = {}
            ctf_event["Proc"] = "Proc" + filename[index_of_proc]
            ctf_event["ph"] = "X"  # complete event type
            ctf_event["ts"] = 0.001 * trace.events.begin[e] # when we started, in ms
            ctf_event["dur"] = 0.001 * (trace.events.end[e] - trace.events.begin[e]) # when we started, in ms
            ctf_event["name"] = trace.event_names[trace.events.type[e]]
            ctf_event["pid"] = trace.events.node_id[e]
            ctf_event["tid"] = trace.events.stream_id[e]

            if trace.events.key[e] != None:
                ctf_event["args"] = trace.events.key[e].decode('utf-8').rstrip('\x00')
                ctf_event["name"] = trace.event_names[trace.events.type[e]]+" keys="+ctf_event["args"]+" "+ctf_event["Proc"]+" pid="+str(ctf_event["pid"])+" tid="+str(ctf_event["tid"])

            # ctf_event["pid"] = trace.events.node_id[e]
            # ctf_event["tid"] = trace.events.stream_id[e]

            ctf_data["traceEvents"].append(ctf_event)
        proc = proc+1


    with open(ctf_filename, "w") as chrome_trace:
        json.dump(ctf_data, chrome_trace)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("usage: pbt_to_ctf.py <pbt file> <ctf file>")

    pbt_files_list=[]
    directory = "/tmp"
    ext=('.prof-zIOexi')
    prefix="bspmm-parsec-trace-2-1"

    #iterate over all files within the directory
    for file in os.listdir(directory):
        if ext in file and file.startswith(prefix):
            # print(file)
            pbt_files_list.append(file)

    # read_pbt(sys.argv[1])
    # it takes argument that it does not use
    pbt_to_ctf(pbt_files_list, sys.argv[2]) #sys.argv[1]
