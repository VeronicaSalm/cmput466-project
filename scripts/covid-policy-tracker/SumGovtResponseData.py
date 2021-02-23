"""
Averages the government response data for all countries, to try to determine the
times when restrictions are at their highest and lowest.
The goal is to use this data to identify the start and end of lockdowns.
"""
import csv, os, sys
from collections import defaultdict

# INPUT FILE
in_fname = "data/timeseries/government_response_index.csv"

# FILTER
# Specify a list of country names to leave out (3rd column of input CSV)
# TODO if desired
filtered = set()

# OUTPUT FILE
dest = "daily_response_sum.csv"


# results, will map all the columns from the original file to the sum
# of the country restrictions for that date
sums = defaultdict(float)
errors = 0
with open(in_fname, "r") as fobj:
    reader = csv.reader(fobj)
    headers = reader.__next__()

    # extract all the data
    for row in reader:
        # the first few columns just have country information, which we skip
        if row[2] in filtered:
            # skip any countries in the filtered set
            # note, not tested yet
            continue

        for k, v in zip(headers[3:], row[3:]):
            try:
                sums[k] += float(v)
            except ValueError:
                print(f"Cound not convert invalid value to float: '{v}'")
                errors += 1

with open(dest, "w") as fdest:
    writer = csv.writer(fdest)
    writer.writerow(["Date", "Global Govt Response (Sum)"])
    for k in headers[3:]:
        writer.writerow([k, sums[k]])

print(f"Done! Results stored to {dest}. Total errors found: {errors}.")
