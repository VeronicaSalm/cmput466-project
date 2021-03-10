### Covid Policy Tracker Scripts

This folder contains two scripts used to summarize Covid-19 lockdown information.

To measure lockdown intensity, we utilize the global government response data from the Covid Policy Tracker repository (https://github.com/OxCGRT/covid-policy-tracker). This is a project from the Blavatnik School of Government which collects systematic information on the measures taken by governments around the world to prevent the spread of Covid-19. This information has been collected over time, so it is possible to know when certain measures were introduced and how long they have remained active globally. The researchers measure 19 different indicators of government response measuring containment and closure (C), economic (E), and health system (H) policies. They also provide four indices that aggregate the data into a single number for each country on a given day:
* overall government response index (all indicators)
* containment and health index (all C and H indicators)
* stringency index (all C indicators, plus H1 which records public information campaigns)
* economic support index (all E indicators)
Our project uses the overall government response index (https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/timeseries/government_response_index.csv) to measure the level of government response in each country. Because we want to analyze only English tweets, we only consider the government response (on each day since the start of the pandemic) for predominantly English-speaking countries (https://www.sheffield.ac.uk/international/english-speaking-countries). Thus, we make the assumption that the majority of the English tweets in our dataset probably originate from predominantly English-speaking countries.    

### sum_govt_response_data.py

This script sums the global response index for all predominantly English-speaking countries for every day since Jan 1, 2020. Note that some cells in the input may be NULL - these are ignored by this script for the purpose of summing.

#### Running Instructions
1. Clone the Covid Policy Tracker repo: https://github.com/OxCGRT/covid-policy-tracker
2. Change the variable `in_fname` (line 15) to the path to the file `government_response_index.csv` from the repo.
3. Run the script using: `python3 sum_govt_response_data.py`
The result CSV will be stored in `csvs/daily_response_sum.csv`, unless the `dest` variable on line 18 is changed. The resulting CSV has two columns:
* Date: the given date in the format DDMmmYYYY (e.g., 01Jan2020)
* Global Govt Response (Sum): the sum of the government response data for all countries on this date

### weekly_totals.py

This script further aggregates the daily government response index sums into weekly sums. It must be run **after** the above script `sum_govt_response_data.py`. The source and destination CSVs can be changed by modifiying `in_fname` and `dest` on lines 12 and 15.      

#### Running Instructions
```
python3 weekly_totals.py
```
The result CSV will be stored in `csvs/weekly_response_sum.csv`, unless the `dest` variable on line 15 is changed. The resulting CSV has three columns:
* Week Number: enumerates the weeks starting from 1
* Date: the given date in the format DDMmmYYYY (e.g., 01Jan2020)
* Global Govt Response (Sum): the sum of the government response data for all countries on this week

### Images
A PNG image has been generated from the resulting data: `images/Week Number, English (Normalized) and All (Normalized).png`, a line plot of the weekly global response sum for every week of the pandemic, comparing the response of English speaking countries to the line of all countries. This image was used to find points of interest for our analysis. An original PNG image (with only the line for all countries) was created first and is also included: `images/Global Govt Response (Sum) vs. Week Number.png`.
