import json
import os
import sys

folder = "follow_results"
cnt = 0
file_id = 0
max_users_per_file = 500
fobj = open(os.path.join(folder, str(file_id)+".json"), "w")
with open("follow_results_backup.json", "r") as f:
    for line in f.readlines():
        cnt += 1 

        if cnt % max_users_per_file == 0:
            fobj.close()
            file_id += 1
            fobj = open(os.path.join(folder, str(file_id)+".json"), "w")
            print(cnt)

        print(line, file=fobj, end="")
