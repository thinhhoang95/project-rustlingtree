# Preparing ADIP resources
1. Download the airport's data from FAA's ADIP: `https://adip.faa.gov/agis/public/#/airportCharts/DFW` as a single `.mhtml` file.
2. Run the script `node src/scenario/adip_resources/extract-kdfw-charts.js data/adip/kdfw_adip_resources.mhtml --out data/adip/kdfw_adip_resources.json` to extract the chart URLs to a single JSON file.
3. Use the script `src/scenario/adip_resources/download_kdfw_charts.py`. Despite its name, you can use any JSON manifest you just created for any airport, and it will automatically download all associated PDF resources to a local directory: `python -m scenario.adip_resources.download_kdfw_charts data/adip/kdfw_adip_resources.json --output-dir data/adip/charts`.
4. Convert all PDF files to JPG files so that the LLM could read them with `python -m scenario.adip_resources.convert_adip_charts_to_jpg data/adip/charts --output-dir data/adip/charts/img`.
5. 