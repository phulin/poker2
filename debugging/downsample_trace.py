#!/usr/bin/env python3
"""
Downsample a Perfetto/Chrome trace file using various methods.

This script reads a Chrome trace format JSON file (compatible with Perfetto)
and downsamples using one of three methods:
- random: Select a random contiguous window of size length/ratio
- ratio: Keep every Nth event (where N=ratio)
- slice: Take the first length/ratio events from the start

Usage:
    python debugging/downsample_trace.py input.json output.json --ratio 10 --method random
    python debugging/downsample_trace.py input.json output.json -r 5 -m ratio
    python debugging/downsample_trace.py input.json output.json -r 10 -m slice
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import orjson


def downsample_trace(input_path: Path, output_path: Path, ratio: int, method: str):
    """
    Downsample a Chrome trace JSON file using the specified method.

    Args:
        input_path: Path to input trace file
        output_path: Path to output trace file
        ratio: Downsample ratio
        method: Downsampling method - 'random', 'ratio', or 'slice'
    """
    print(f"Reading trace from: {input_path}")
    with open(input_path, "rb") as f:
        trace_data = orjson.loads(f.read())

    # Chrome trace format is typically a list of events
    if isinstance(trace_data, list):
        events = trace_data
    elif isinstance(trace_data, dict):
        # Some traces have events in a 'traceEvents' key
        events = trace_data.get("traceEvents", trace_data.get("events", []))
        if isinstance(events, dict):
            events = list(events.values())
    else:
        raise ValueError(f"Unexpected trace format: {type(trace_data)}")

    total_events = len(events)
    print(f"Total events: {total_events}")

    # Remove all events where event["cat"] == "cpu_op"
    # events = [
    #     event
    #     for event in events
    #     if event.get("cat") != "cpu_op" and event.get("cat") != "cpu_instant_event"
    # ]
    total_events = len(events)

    # Apply downsampling based on method
    if method == "window":
        # Calculate window size
        window_size = total_events // ratio
        if window_size == 0:
            raise ValueError(f"Ratio {ratio} is too large. Window size would be 0.")

        # Pick a random start position
        max_start = total_events - window_size
        start_pos = random.randint(0, max_start)
        end_pos = start_pos + window_size

        # Extract the window
        downsampled_events = events[start_pos:end_pos]
        downsampled_count = len(downsampled_events)

        print(
            f"Downsampling with ratio {ratio} (method=random): "
            f"selected random window [{start_pos}:{end_pos}] "
            f"({downsampled_count} events, {100*downsampled_count/total_events:.1f}%)"
        )

    elif method == "ratio":
        # Keep every Nth event
        downsampled_events = events[::ratio]
        downsampled_count = len(downsampled_events)

        print(
            f"Downsampling with ratio {ratio} (method=ratio): "
            f"keeping every {ratio}th event "
            f"({downsampled_count} events, {100*downsampled_count/total_events:.1f}%)"
        )

    elif method == "slice":
        # Take first N events
        window_size = total_events // ratio
        if window_size == 0:
            raise ValueError(f"Ratio {ratio} is too large. Window size would be 0.")

        downsampled_events = events[:window_size]
        downsampled_count = len(downsampled_events)

        print(
            f"Downsampling with ratio {ratio} (method=slice): "
            f"taking first {window_size} events "
            f"({downsampled_count} events, {100*downsampled_count/total_events:.1f}%)"
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be 'random', 'ratio', or 'slice'"
        )

    # Reconstruct trace structure
    if isinstance(trace_data, list):
        output_data = downsampled_events
    elif isinstance(trace_data, dict):
        output_data = trace_data.copy()
        if "traceEvents" in output_data:
            output_data["traceEvents"] = downsampled_events
        elif "events" in output_data:
            output_data["events"] = downsampled_events

    print(f"Writing downsampled trace to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(output_data))

    print(f"Done! Reduced from {total_events} to {downsampled_count} events")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a Perfetto/Chrome trace file by a specified ratio"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input trace file (JSON format)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output trace file path",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=int,
        required=True,
        help="Downsample ratio",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["window", "ratio", "slice"],
        default="window",
        help="Downsampling method: 'window' (random window), 'ratio' (every Nth event), "
        "'slice' (first N events). Default: window",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file does not exist: {args.input}")

    if args.ratio < 1:
        parser.error("Ratio must be >= 1")

    downsample_trace(args.input, args.output, args.ratio, args.method)


if __name__ == "__main__":
    main()
