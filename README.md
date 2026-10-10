.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

python -m streamlit run app.py



















# Hospital Multi-Format Data Integration, EDA and Preprocessing

A local Streamlit application based on `Hospital_Multi_Format_EDA_Preprocessing_Full(3).ipynb`.

## Features

- Upload one or more CSV, JSON, XML, HTML, KML and KMZ files.
- Parse files using format-specific readers.
- Standardize records to the notebook's common 15-field hospital schema.
- Combine datasets and show per-file record counts.
- Explore missing values, data types, distributions, descriptive statistics and IQR outliers.
- Apply notebook-aligned preprocessing: trim text, normalize common categories, numeric conversion, invalid-value handling, coordinate checks, ZIP and phone validation, duplicate removal, and median imputation for `Bed_Count` and `Quality_Score`.
- View validation checks and download standardized/cleaned CSVs.
- Download a formatted Excel workbook with data and reports.

## Run locally (Windows)

1. Install Python 3.10 or later.
2. Extract this project folder.
3. Open Command Prompt or PowerShell in the extracted folder.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the website:

   ```powershell
   streamlit run app.py
   ```

7. Open the local address Streamlit prints, usually `http://localhost:8501`.

## Run locally (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Use the app

1. Use the sidebar to upload one or more supported files.
2. Select **Process uploaded files**.
3. Review the Overview, Integrated data, EDA, Preprocessing and Validation tabs.
4. Open Export and download the Excel or CSV output.

## Schema mapping notes

The source notebook uses these format-specific source fields:
- CSV: `Hospital_Name`, `Address`, `City`, `ZIP_Code`, `Phone`, `Website`, `County_Status`, `Longitude`, `Latitude`, `Location`, `Data_Year`, `Hospital_Type`, `Bed_Count`, `Inspection_Score`.
- JSON: `hospital_name`, `street_address`, `city`, `postal_code`, `telephone`, `web`, `county_status`, `lon`, `lat`, `year`, `hospital_type`, `beds`, `rating`.
- XML: `HospitalName`, `Street`, `Town`, `PostalCode`, `ContactPhone`, `Website`, `CountyStatus`, `Longitude`, `Latitude`, `Year`, `Type`, `Beds`, `QualityScore`.
- HTML: `Facility Name`, `Street Address`, `Municipality`, `Postal`, `Contact`, `Web Link`, `County Flag`, `Lon`, `Lat`, `Reporting Year`, `Facility Category`, `Beds Available`, `Record Status`.
- KML/KMZ: Placemark `name`, `description`, and `coordinates`.

A normalized alias fallback is included to support common variations in uploaded headers. When a field does not exist in a source, it is left missing rather than invented.

## Important behavior

- This app processes files in memory during the local session; it does not write uploaded source files to disk.
- The app does not silently fabricate descriptive values such as phone numbers or addresses.
- The notebook's source mapping and cleaning policy are preserved where applicable. The website also adds robust handling for some user-uploaded schema variations.
- KML/KMZ records naturally have fewer fields than tabular hospital datasets, so those columns will be missing unless supplied in the source.
- The Excel report has formatted headers, frozen top rows and highlighted `NA` cells.

## Troubleshooting

- If `streamlit` is not recognized, ensure the virtual environment is activated and run `python -m streamlit run app.py`.
- If an HTML upload fails, check that the file contains at least one readable HTML table.
- If XML parsing returns zero records, verify that repeated records are represented as `Hospital`, `Record`, or `Facility` elements.
- If a column is not mapped, check its spelling against the schema mapping notes. The app's EDA tab and mapping CSV show which fields were mapped.
