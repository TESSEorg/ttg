#!/usr/bin/env python3

import otf2
from otf2.events import *

def read_trace(trace_name="ttg.otf2"):
    with otf2.reader.open(trace_name) as trace:
        print("Read {} string definitions".format(len(trace.definitions.strings)))

        for loc in trace.definitions.locations:
            print("Location definition with value '{}' in trace.".format(loc))

        for str in trace.definitions.strings:
            print("String definition with value '{}' in trace.".format(str))

        print("Read {} events".format(len(trace.events)))

        for location, event in trace.events:
            if isinstance(event, Enter):
                print("Encountered enter event into '{}' on location {} at {}".format(event.region.name, location.name, event.time))
            elif isinstance(event, Leave):
                print("Encountered leave event for '{}' on location {} at {}".format(event.region.name, location.name, event.time))
            else:
                print("Encountered event on location {} at {}".format(location.name, event.time))

import json
import re
import sys

def parsec_otf2_to_chrome(otf2_trace_name="ttg.otf2", chrome_trace_name="ttg.ctf.json"):
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
    #read_trace(sys.argv[1])
    if len(sys.argv) == 1:
        sys.exit("usage: otf2_to_ctf.py <otf2 trace file> <ctf trace file>")
    parsec_otf2_to_chrome(sys.argv[1], sys.argv[2])
