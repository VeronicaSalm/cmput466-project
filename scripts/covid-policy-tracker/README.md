### Covid Policy Tracker Scripts

This folder contains two scripts used to summarize covid 19 lockdown info, so we can determine time periods of interest in the data.    
* SumGovtResponseData.py must be copied into the main folder of the Covid Policy Tracker repo (https://github.com/OxCGRT/covid-policy-tracker) before it can be run using "python3 SumGovtResponseData.py". It sums the global response index for all countries for every day since Jan 1, 2020 (see the linked repo for information on what this is). Note that some cells in the input are NULL - these are ignored by my script for the purpose of summing.
* WeeklyTotals.py must be run after SumGovtResponseData.py, in the same folder as the output of that script. It creates a weekly sum for every week of the pandemic.    

The output CSVs of each script and a PNG of a plot made using the weekly_response_sum.csv data are also included.
