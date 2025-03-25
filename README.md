RecessionPredictor
This is a Python tool that predicts U.S. recession probabilities for 3 months, 6 months, and 12 months using economic data from the FRED API and a Random Forest Classifier.
Overview
The script pulls data like GDP, yield spread, unemployment rate, CPI, and NASDAQ returns from FRED, processes it, and trains models to forecast recession odds. It outputs classification reports, the latest probabilities, and separate plots for each time horizon showing trends with recession periods shaded. It’s set to run as of March 25, 2025, with data up to February 28, 2025.
Features
Uses FRED data: GDP, T10Y2Y, UNRATE, CPIAUCSL, USREC, NASDAQCOM

Predicts recessions for 3, 6, and 12 months

Random Forest Classifier with balanced weights

Plots probability trends and feature trends

Example output from last run:
Predicted probabilities from 2025-02-28:
  Recession within 3 months: 8.00%
  Recession within 6 months: 4.00%
  Recession within 12 months: 0.00%

Prerequisites
Python 3.8 or higher

A FRED API key (get one from fred.stlouisfed.org)

Installation
Clone the repo:
git clone https://github.com/yourusername/RecessionPredictor.git
cd RecessionPredictor

Set up a virtual environment (optional):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Add your FRED API key:
Open Recession.py and replace 'YOURKEYHERE' with your key:
fred = Fred(api_key='your_api_key_here')

Requirements
Create a requirements.txt file with:
pandas
fredapi
scikit-learn
matplotlib
numpy
Then run:
pip install pandas fredapi scikit-learn matplotlib numpy
Usage
Run the script:
python Recession.py

Output:
Console shows classification reports and latest probabilities

Plots show 3M, 6M, and 12M probabilities plus feature trends

Customize:
Change 'today' in the script to adjust the analysis date (default: March 25, 2025)

Edit 'features' list for different indicators

Update 'series_ids' for other FRED data

File Structure
Recession.py: Main script for data, models, and plots

requirements.txt: List of dependencies

README.md: This file

Notes
NASDAQCOM starts in 1971, so historical data is limited compared to GDP (1947). For more history, try yfinance for S&P 500 (see code comments).

A SettingWithCopyWarning might pop up from Pandas slicing; it’s fixed with .copy() on df_full.

Possible upgrades: add feature importance, use multi-output models, or tap other data sources.

License
MIT License - see LICENSE file if included.
Contributing
Submit issues or pull requests if you’d like. Ideas for features or data are appreciated.
Acknowledgments
Data from FRED (fred.stlouisfed.org)

Built from economic forecasting and machine learning concepts

