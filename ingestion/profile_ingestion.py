import pstats
p = pstats.Stats("profile_output.prof")
p.sort_stats("cumulative").print_stats(50)  # top 20 by cumulative time

"""analyze previously exported file with command
python -m cProfile -o profile_output.prof -m ingestion.ingest --source local
python -m cProfile -o profile_output.prof -m ingestion.ingest --source gitlab

start in bg with 
 #cmd /c "start /B python -m ingestion.ingest --source gitlab > ingest_log.txt 2>&1"

 control log with
# powershell Get-Content ingest_log.txt -Wait -Tail 20

monitor with 
#tasklist /v | findstr python

 kill with
#taskkill /PID 24688 /F
"""

