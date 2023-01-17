# Time Series Anonymization with (k,P)-Anonymity

## Requirements
- numpy
- pandas
- loguru
- saxpy
- matplotlib

You can use the requirements.txt file
```console
pip3 install -r requirements.txt
```

## Usage
For a single run:
```console
python3 kp-anonymity.py k_value p_value paa_value /where/is/the/dataset.csv
```
>### Parameters
> - k_value: Value of k-anonymity (greater than p value)
> - p_value: Value of the pattern, p-anonymity
> - paa_value: Length of the string that can be used to describe the pattern
> - /where/is/the/dataset.csv: Path of the input dataset

For run the tests and produce the histograms of execution time and instant value loss for each run:
```console
python3 runTests.py
```
> You can find the histograms inside "Output" folder as .png files.\
> The terminal output informs about:
> - The parameters that give the lowest "execution time"
> - The parameters that give the lowest "instant value loss".

## Datasets
In "Dataset" folder there are two datasets that can be use as input dataset. The input datasets are .csv files.
- [MLTollsStackOverflow.csv](./Dataset/MLTollsStackOverflow.csv)
- [NIFTY50-1_day_with_indicators.csv](./Dataset/NIFTY50-1_day_with_indicators.csv)
- [Sales_Transaction_Dataset_Weekly_Final.csv](./Dataset/Sales_Transaction_Dataset_Weekly_Final.csv)

## Presentation
Read the presentation: [(k,P)-Anonymity](./Presentation/k_P-Anonymity.pdf)

---
Work done by:
* [Alex Di Stefano](https://github.com/xDiste)
* [Christian Stingone](https://github.com/ChriStingo)

