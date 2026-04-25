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
The script to realize this is `src/scenario/demand_opensky/1_1download_ostrino.py`. Make sure you have ostrino CLI ready in the project root.

### ADS-B Data Processing
