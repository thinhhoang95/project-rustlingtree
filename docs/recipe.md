# Downloading Airport Charts Automatically
> Note that we will use CIFP for procedure extraction, but these charts will provide visual confirmation.

1. Download the airport's data from FAA's ADIP: `https://adip.faa.gov/agis/public/#/airportCharts/DFW` as a single `.mhtml` file.
2. Run the script `node src/scenario/adip_resources/extract-kdfw-charts.js data/adip/kdfw_adip_resources.mhtml --out data/adip/kdfw_adip_resources.json` to extract the chart URLs to a single JSON file.
3. Use the script `src/scenario/adip_resources/download_kdfw_charts.py`. Despite its name, you can use any JSON manifest you just created for any airport, and it will automatically download all associated PDF resources to a local directory: `python -m scenario.adip_resources.download_kdfw_charts data/adip/kdfw_adip_resources.json --output-dir data/adip/charts`.
4. Convert all PDF files to JPG files so that the LLM could read them with `python -m scenario.adip_resources.convert_adip_charts_to_jpg data/adip/charts --output-dir data/adip/charts/img`. --> this step is unreliable, you should not do it, use the CIFP procedure below.

# Extraction of CIFP Resources
1. Download CIFP data from FAA. Put it in data/cifp, rename to FAACIFP18.txt (add .txt extension).
2. Run the notebook `src/scenario/cifp_parser/parse.ipynb`, it will produce `data/kdfw_procs/` procedure CSV files.

If you want to know what the fields mean, check out `docs/cifp/arinc424_route_and_section_code_reference.md`.

# ADS-B Demand Data Download
The script to realize this is `src/scenario/demand_opensky/1_1download_ostrino.py`. Make sure you have ostrino CLI ready in the project root. This will give `data/adsb/raw` CSV files. You need to provide the datetime range of data to be downloaded, as well as the timezone.

Then you can run the script `src/scenario/demand_opensky/extract_departures_and_arrivals.py` to automatically extract takeoffs and landings. Note that it will also download an authoritative departures and arrivals catalog from OpenSky to cross-validate the data validity. You must supply the same datetime range and timezone as in the ostrino download step.

### ADS-B Data Processing
Generate the arrival, departure catalogs, and compress data for the rustlingleaves client.

### Arrival and Departure Catalog
python src/scenario/demand_opensky/extract_departures_and_arrivals.py \
  --from-datetime "2025-04-01T00:00:00" \
  --to-datetime "2025-04-01T23:59:59" \
  --timezone "America/Chicago" \
  --split-gap-seconds 1500

This step writes the derived landings/departures catalog, the fix-sequence catalog, and the authoritative departures/arrivals catalog into `data/adsb/catalogs`. You might want to check that the derived departure/arrival catalog should not miss too many flights (possibly less than 1-2%) of the OpenSky's authoritative catalog in the console output.

### Compress the Data
python src/scenario/trajectory_compressor/cli.py --landings-departures-catalog

This writes the full ADS-B compressed trajectory file used for departures at `data/adsb/compressed/adsb_compressed_flights.jsonl`. SIMAP arrival artifacts are written separately by `scenario-manager-precompute-artifact` to `data/artifacts/simap_arrival_flights.jsonl`.

# Hail Mary
## 1. Building ADS-B trajectory corpus 
A trajectory corpus is a set of observed ADS-B trajectories that will be used to scale traffic demand in Hail Mary scenarios. The corpus only contains trajectories that are considered to be valid, such as terminating "properly" at the runway threshold (the exact definition is quite nuanced to account for edge cases like flight number and icao24 are designated for both arrival and imminent departure) For example: if a window `00:20-01:20` has 30 traffic counts, then a scale of 1.1 will create 3 additional traffic counts. That means that 3 flights will be pulled from the corpus for the corresponding (runway arrival) cluster.

To build the trajectory corpus, run the following command:
```bash
OUT=data/artifacts/hailmary/corpus
./.venv/bin/python -m hailmary.cli.build_offline_corpus \
  --manifest data_manifest.json --airport KDFW --output-dir "$OUT"
```