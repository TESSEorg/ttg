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

#giving directory name
dirname = "/tmp"

prefix="bspmm-parsec-trace-2048-256"

#giving file extension
ext=('.prof')

list_of_files=[]

#iterating over all files within the specified directory
for file in os.listdir(dirname):
    if ext in file and file.startswith(prefix):
        # print(file)
        list_of_files.append(file)
print(list_of_files)

# output_file= list_of_files[0]

with open('OutputFile.bin', 'wb') as f:
    for file in list_of_files:
        with open(file,'rb') as s:
            f.write(s.read())
print(f)


def read_pbt(pbt_filename):
    ptt_filename = pbt2ptt.convert([pbt_filename], multiprocess=False)
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

def pbt_to_ctf(pbt_filename, ctf_filename):
    ctf_data = {"traceEvents": []}
    ptt_filename = pbt2ptt.convert([pbt_filename], multiprocess=False)
    trace = ptt.from_hdf(ptt_filename)

    for e in range(len(trace.events)):
        # print('id=',trace.events.id[e],' node_id=',trace.events.node_id[e],' stream_id=',trace.events.stream_id[e],'key=',trace.events.key[e],' type=',trace.events.type[e],' b=',trace.events.begin[e],' e=',trace.events.end[e])
        # print('\n')
        ctf_event = {}
        ctf_event["ph"] = "X"  # complete event type
        ctf_event["ts"] = 0.001 * trace.events.begin[e] # when we started, in ms
        ctf_event["dur"] = 0.001 * (trace.events.end[e] - trace.events.begin[e]) # when we started, in ms
        ctf_event["name"] = trace.event_names[trace.events.type[e]]
        if trace.events.key[e] != None:
            # ctf_event["args"] = trace.events.key[e].decode("utf-8")
            ctf_event["args"] = trace.events.key[e].decode('utf-8').rstrip('\x00')
            ctf_event["name"] = trace.event_names[trace.events.type[e]]+ctf_event["args"]

        ctf_event["pid"] = trace.events.node_id[e]
        ctf_event["tid"] = trace.events.stream_id[e]

        ctf_data["traceEvents"].append(ctf_event)

    with open(ctf_filename, "w") as chrome_trace:
        json.dump(ctf_data, chrome_trace)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("usage: pbt_to_ctf.py <pbt file> <ctf file>")
    # read_pbt(sys.argv[1])
    pbt_to_ctf(sys.argv[1], sys.argv[2])
