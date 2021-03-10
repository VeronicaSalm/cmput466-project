# -----------------------------------------------------------------------------
# weekly_totals.py
#
# Converts the government response data (daily sums) into weekly sums,
# starting with Monday of each week.
# -----------------------------------------------------------------------------

import csv, os, sys
from collections import defaultdict

# INPUT FILE
in_fname = "csvs/daily_response_sum.csv"

# OUTPUT FILE
dest = "csvs/weekly_response_sum.csv"


if __name__ == "__main__":
    # results, will map all the columns from the original file to the sum
    # of the country restrictions for that date
    sums = defaultdict(float)
    headers = []
    with open(in_fname, "r") as fobj:
        reader = csv.reader(fobj)

        # ignore headers
        reader.__next__()

        rows = list(reader)

        # first week is short, since Jan 1 2020 was a Wednesday
        week = rows[0][0] + "-" + rows[4][0]
        sums[week] = sum([float(rows[i][1]) for i in range(0,5)])
        headers.append(week)

        # w is an initial offset, needed because the first week is only 5 days
        w = 5
        WEEK_LEN = 7
        # NOTE: This ignores a partial week at the end, but this shouldn't
        # matter for our purposes (we will likely not use such recent data anyway)
        while (w + WEEK_LEN < len(rows)):
            week = rows[w][0] + "-" + rows[w+WEEK_LEN-1][0]
            sums[week] = sum([float(rows[i][1]) for i in range(w,w+WEEK_LEN-1)])
            headers.append(week)
            w += WEEK_LEN

    # store the sum for each week in the output file
    with open(dest, "w") as fdest:
        writer = csv.writer(fdest)
        # week numbers start at 1 for convenience when plotting
        writer.writerow(["Week Number","Week", "Global Govt Response (Sum)"])
        for i in range(len(headers)):
            k = headers[i]
            writer.writerow([i+1, k, sums[k]])

    print(f"Done! Results stored to {dest}.")
