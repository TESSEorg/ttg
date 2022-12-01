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

def read_pbt(pbt_filename):
    ptt_filename = pbt2ptt.convert([pbt_filename], multiprocess=False)
    trace = ptt.from_hdf(ptt_filename)

    print('The columns of the DataFrame (or data labels) and their datatypes are:')
    print(trace.events.dtypes)

    print('There are ' + str(len(trace.events)) + ' events in this trace', end=' ')
    for e in range(len(trace.events)):
        print('id=',trace.events.id[e],' key=',trace.events.key[e],' type=',trace.events.type[e],' b=',trace.events.begin[e],' e=',trace.events.end[e],)

import json
import re
import sys

def pbt_to_ctf(otf2_trace_name, chrome_trace_name):
    chrome_data = {"traceEvents": []}
    with otf2.reader.open(otf2_trace_name) as trace:
        for location, event in trace.events:
            if not event.region.name.startswith("PARSEC "):
                ctf_event = {}
                ctf_event["name"] = event.region.name
                ctf_event["ts"] = event.time / 1000  # convert ns to µs

                # location.name = "PaRSEC Thread ThreadId of VP ProcessorId"
                loc_match = re.match("PaRSEC Thread (\d+) of VP (\d+)", location.name)
                if loc_match == None:
                    sys.exit("unexpected format for event's location.name")
                ctf_event["pid"] = loc_match[2]
                ctf_event["tid"] = loc_match[1]

                # event type
                if isinstance(event, Enter):
                    ctf_event["ph"] = "B"
                elif isinstance(event, Leave):
                    ctf_event["ph"] = "E"
                chrome_data["traceEvents"].append(ctf_event)

    with open(chrome_trace_name, "w") as chrome_trace:
        json.dump(chrome_data, chrome_trace)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("usage: pbt_to_ctf.py <pbt file> <ctf file>")
    read_pbt(sys.argv[1])
    #pbt_to_ctf(sys.argv[1], sys.argv[2])
