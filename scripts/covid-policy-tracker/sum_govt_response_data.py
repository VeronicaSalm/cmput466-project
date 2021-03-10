# -----------------------------------------------------------------------------
# sum_govt_response_data.py
#
# Averages the government response data for all countries, to try to determine
# the times when restrictions are at their highest and lowest.
# The goal is to use this data to identify the start and end of lockdowns.
# -----------------------------------------------------------------------------

import csv
import os
import sys
from collections import defaultdict

# INPUT FILE -- PATH TO RESPONSE DATA
in_fname = "data/timeseries/government_response_index.csv"

# OUTPUT FILE
dest = "csvs/daily_response_sum.csv"

# FILTER ENGLISH COUNTRIES
# Specify a list of country names to include (3rd column of input CSV)
# taken from the list of countries which have English as an official language:
# https://www.sheffield.ac.uk/international/english-speaking-countries
# note: countries were removed if they did not appear in the input CSV as countries
english_countries = {
            "Australia",
            "Bahamas",
            "Barbados",
            "Belize",
            "Canada",
            "Dominica",
            "Guyana",
            "Ireland",
            "Jamaica",
            "Malta",
            "New Zealand",
            "Trinidad and Tobago",
            "United Kingdom",
            "United States"}


if __name__ == "__main__":
    # results, will map all the columns from the original file to the sum
    # of the country restrictions for that date
    sums = defaultdict(float)
    errors = 0
    with open(in_fname, "r") as fobj:
        reader = csv.reader(fobj)

        # extract the headers
        headers = reader.__next__()

        for row in reader:
            # only consider countries in the English-speaking list
            if row[2] not in english_countries:
                continue

            # the first few columns just have country information, which we skip
            # (this is where the [3:] comes from)
            for k, v in zip(headers[3:], row[3:]):
                try:
                    sums[k] += float(v)
                except ValueError:
                    print(f"Could not convert invalid value to float: '{v}'")
                    errors += 1

    with open(dest, "w") as fdest:
        writer = csv.writer(fdest)
        writer.writerow(["Date", "Global Govt Response (Sum)"])
        for k in headers[3:]:
            writer.writerow([k, sums[k]])

    print(f"Done! Results stored to {dest}. Total errors found: {errors}.")
