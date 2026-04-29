import subprocess
import datetime
import requests
import os
from multiprocessing import Pool, cpu_count, Value, Manager

# jwt = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ0SVIwSDB0bmNEZTlKYmp4dFctWEtqZ0RYSWExNnR5eU5DWHJxUzJQNkRjIn0.eyJleHAiOjE3MjUzNTY4NjUsImlhdCI6MTcyNTM0OTY2NSwianRpIjoiMGNiYzYxZTEtNDI2Mi00MDA3LTg5MTQtZTgxN2EzNjRmM2M5IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLm9wZW5za3ktbmV0d29yay5vcmcvYXV0aC9yZWFsbXMvb3BlbnNreS1uZXR3b3JrIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjEzYmYwYmQwLTMzOTktNDA2NS04ZGFiLTIyYzI0Njg1N2E4MSIsInR5cCI6IkJlYXJlciIsImF6cCI6InRyaW5vLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiIxZjQ2MzhhMy0wYjk4LTRmYzctOWNlOC1jZDBiOWJiMGI1N2UiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtb3BlbnNreS1uZXR3b3JrIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGdyb3VwcyBlbWFpbCIsInNpZCI6IjFmNDYzOGEzLTBiOTgtNGZjNy05Y2U4LWNkMGI5YmIwYjU3ZSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiVGhpbmggSG9hbmciLCJncm91cHMiOlsiL29wZW5za3kvdHJpbm8vcmVhZG9ubHkiXSwicHJlZmVycmVkX3VzZXJuYW1lIjoidGhpbmhob2FuZ2RpbmgiLCJnaXZlbl9uYW1lIjoiVGhpbmgiLCJmYW1pbHlfbmFtZSI6IkhvYW5nIiwiZW1haWwiOiJ0aGluaC5ob2FuZ2RpbmhAZW5hYy5mciJ9.GeiFz_Num5kgHA6BwosqilwlKUmdpi8fKsaPIO06PotBWLmCkrl7Y6g-os60xyILuvd31W1T-pT0-llwTPyO1PBs0VsCsOPa1mgrRyFE5uAa-QFGV_MuaLcd6BGeTR6Ss9E2vrJYcmNu630uKYu3UJOYjeCA0whkUccAUKiBiGrQohwlec0Ryz1I67rEruENt6sgV3urrywURJ8BDtJPbMnqdrG_FpMgqaWl83PEsN2aypL9Oq36fOOT68gZONgvx5s1SU6SUIDKzEVRL_V8JhBzDaY8fWJNuHaZYAsPTyUoOV_ChUSIeyvek5nm8BFsH8fjc-tUvfSXXgpZUd5jpg'

MASTER_DATASET_PREFIX = 'data/adsb'

# # Resolve the Trino CLI path relative to this file so we don't depend on $PATH.
# # This assumes the `trino` executable lives at the project root alongside `trino.jar`.
TRINO_BIN = "/Volumes/CrucialX/project-rustlingtree/trino"

print("Resolved Trino CLI path: ", TRINO_BIN)

def get_jwt():
    result = requests.post(
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
    data={
        "client_id": "trino-client",
        "grant_type": "password",
        "username": "thinhhoangdinh",
        "password": "iQ6^yrwe7o3m",
        }
    )
    print('Obtained JWT: ', result.json()['access_token'][:10])
    return result.json()['access_token']

def init_worker(counter, jwt_holder, total_files_):
    global downloaded_counter, shared_jwt, total_files
    downloaded_counter = counter
    shared_jwt = jwt_holder
    total_files = total_files_

def download_for_timestamp(timestamp):
    """
    Downloads data for a single timestamp.
    """
    global downloaded_counter, shared_jwt
    
    # Check if JWT needs to be refreshed (every 240 files)
    with downloaded_counter.get_lock():
        if downloaded_counter.value > 0 and downloaded_counter.value % 240 == 0:
            new_jwt = get_jwt()
            shared_jwt['token'] = new_jwt
    
    # Get current JWT
    current_jwt = shared_jwt['token']
    
    # Check if the file already exists (legacy flat layout) and is non‑empty.
    # If the file exists but is empty, we treat it as a failed/partial download
    # and will re‑download instead of skipping.
    legacy_path = f"{MASTER_DATASET_PREFIX}/raw/{timestamp}.csv"
    if os.path.exists(legacy_path):
        size = os.path.getsize(legacy_path)
        if size > 0:
            with downloaded_counter.get_lock():
                downloaded_counter.value += 1
                current = downloaded_counter.value
                print(f"File {timestamp}.csv already exists (legacy layout, size={size} bytes). Skipping download. Progress: {current/total_files:.1%}")
            return
        else:
            print(f"WARNING: Legacy file {timestamp}.csv exists but is empty (size=0). Will re-download.")
    
    timestamp_dt = datetime.datetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc)
    print("Current timestamp: ", int(timestamp))
    print("Current datetime: ", timestamp_dt.strftime('%Y-%m-%d %H:%M:%S'))
    # Get the date of the timestamp in YYYY-MM-DD format, always in UTC.
    date = timestamp_dt.strftime('%Y-%m-%d')

    # Create the date folder if it doesn't exist
    os.makedirs(f'{MASTER_DATASET_PREFIX}/raw/{date}', exist_ok=True)

    output_path = f"{MASTER_DATASET_PREFIX}/raw/{date}/{timestamp}.csv"

    # Check if the file already exists and is non‑empty.
    # If it's empty, we will re‑download instead of skipping so that
    # failed downloads do not leave behind empty CSVs.
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 0:
            with downloaded_counter.get_lock():
                downloaded_counter.value += 1
                current = downloaded_counter.value
                print(f"File {timestamp}.csv already exists (size={size} bytes). Skipping download. Progress: {current/total_files:.1%}")
            return
        else:
            print(f"WARNING: Existing file {timestamp}.csv is empty (size=0). Will re-download.")
    
    command = f"{TRINO_BIN} --user=thinhhoangdinh --server=https://trino.opensky-network.org --access-token={current_jwt} --catalog 'minio' --schema 'osky' --execute='SELECT \
        v.time, v.icao24, v.lat, v.lon, v.heading, v.callsign, v.geoaltitude \
    FROM \
        state_vectors_data4 v \
    JOIN ( \
        SELECT \
            FLOOR(time / 60) AS minute, \
            MAX(time) AS recent_time \
        FROM \
            state_vectors_data4 \
        WHERE \
            hour = {timestamp} \
        GROUP BY \
            FLOOR(time / 60) \
    ) AS m \
    ON \
        FLOOR(v.time / 60) = m.minute \
        AND v.time = m.recent_time \
    WHERE \
        v.lat BETWEEN 29.65 AND 36.14 \
        AND v.lon BETWEEN -100.91 AND -93.17 \
        AND v.hour = {timestamp} \
        AND v.time - v.lastcontact <= 15;' --output-format CSV > {output_path}"

    # Run Trino and capture stderr so we can see if/why it failed.
    result = subprocess.run(
        command,
        shell=True,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        # If the command failed, log the error and clean up any empty file Trino may have created.
        print(f"ERROR: Trino query failed for timestamp {timestamp} with return code {result.returncode}")
        if result.stderr:
            print("Trino stderr:")
            print(result.stderr.strip())

        if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
            os.remove(output_path)
            print(f"Removed empty output file {output_path} created by failed query.")
        return

    # At this point the command exited successfully. Double‑check that the file is non‑empty.
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"WARNING: Trino reported success but output file {output_path} is missing or empty.")
        return

    with downloaded_counter.get_lock():
        downloaded_counter.value += 1
        current = downloaded_counter.value
        print(f"Downloaded {timestamp}.csv (size={os.path.getsize(output_path)} bytes). Progress: {current/total_files:.1%}")

def execute_trino_commands(from_datetime, to_datetime):
    """
    Executes Trino shell commands in parallel for each hour within the specified datetime range.
    """
    # Generate UTC timestamps directly so we never depend on pandas' integer view
    # or the local machine timezone.
    if from_datetime.tzinfo is None:
        from_datetime = from_datetime.replace(tzinfo=datetime.timezone.utc)
    else:
        from_datetime = from_datetime.astimezone(datetime.timezone.utc)

    if to_datetime.tzinfo is None:
        to_datetime = to_datetime.replace(tzinfo=datetime.timezone.utc)
    else:
        to_datetime = to_datetime.astimezone(datetime.timezone.utc)

    hourly_timestamps = []
    current_datetime = from_datetime
    while current_datetime <= to_datetime:
        hourly_timestamps.append(int(current_datetime.timestamp()))
        current_datetime += datetime.timedelta(hours=1)
    
    global total_files
    total_files = len(hourly_timestamps)
    print(f'There are {total_files} splits to download')
    
    # Use only one process to avoid stressing the server.
    num_processes = 1
    print(f"Using {num_processes} processes for parallel downloads")
    
    # Create a shared counter and JWT holder
    counter = Value('i', 0)
    manager = Manager()
    jwt_holder = manager.dict()
    jwt_holder['token'] = get_jwt()  # Initial JWT
    
    # Create a pool of workers and map the download function to timestamps
    with Pool(processes=num_processes, initializer=init_worker, initargs=(counter, jwt_holder, total_files,)) as pool:
        pool.map(download_for_timestamp, hourly_timestamps)


if __name__ == "__main__":
    from_datetime = datetime.datetime(2025, 4, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    to_datetime = datetime.datetime(2025, 4, 1, 23, 0, 0, tzinfo=datetime.timezone.utc)

    # Create the MASTER download folder
    os.makedirs(MASTER_DATASET_PREFIX, exist_ok=True)

    # Inside MASTER folder, create a raw folder
    os.makedirs(MASTER_DATASET_PREFIX + '/raw', exist_ok=True)

    execute_trino_commands(from_datetime, to_datetime)
