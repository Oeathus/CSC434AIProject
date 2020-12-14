import re
import os
import pandas as pd

og_data_folder = "og_data" + os.path.sep

# List of Raw Data
files = ["Rochester1StationHourlyRawData.txt",
         "Buffalo2StationHourlyRawData.txt",
         "Syracuse2StationHourlyRawData.txt",
         "Rochester1Station2020HourlyRawData.txt",
         "Buffalo2Station2020HourlyRawData.txt",
         "Syracuse2Station2020HourlyRawData.txt"]

# Array to hold Cleaned file names
cleaned_files = []

# Dicts for merging
data_dict = {
    "main": {},
    "2020": {}
}

# Foreach raw data file, fix empty spaces, date, and change column name
for rawDataFileName in files:
    print("Cleaning " + rawDataFileName)
    rawDataFileNameComponents = rawDataFileName.split(".")
    rawDataFileNameComponents[-2] = rawDataFileNameComponents[-2] + "Cleaned"
    cleanedFileName = ".".join(rawDataFileNameComponents)
    cleaned_files.append(cleanedFileName)

    dateFixer = re.compile(r'(\d{4}-\d{2}-\d{2})\s\d{2}:\d{2}')
    spaceRemover = re.compile(r'(?<=,)\s{3}(?=,)')

    rawDataFile = open(og_data_folder + rawDataFileName, 'r')
    rawData = rawDataFile.read()
    rawDataFile.close()

    cleanedData = dateFixer.sub(r'\1', rawData)
    cleanedData = spaceRemover.sub("", cleanedData)
    cleanedData = re.sub("valid", "day", cleanedData)

    cleanedDataFile = open(og_data_folder + cleanedFileName, 'w')
    cleanedDataFile.write(cleanedData)
    cleanedDataFile.close()
    print("Cleaned " + rawDataFileName + " to " + cleanedFileName)

# Once cleaned, the data needs to be squashed to daily from hourly
for file in cleaned_files:
    print("Averaging " + file)
    data = pd.read_csv(og_data_folder + file, low_memory=False)
    data.drop(data[data['vsby'] < 0].index, inplace=True)
    groupedData = data.groupby('day').agg({'p01i': 'sum', 'tmpf': 'mean',
                                           'dwpf': 'mean', 'relh': 'mean',
                                           'drct': 'mean', 'sknt': 'mean',
                                           'alti': 'mean', 'mslp': 'mean',
                                           'vsby': 'mean'})

    fileNameComponents = file.split(".")
    newFileName = fileNameComponents[0].replace("RawDataCleaned", "AveragedData") + ".csv"
    groupedData.to_csv(og_data_folder + newFileName)

    target_dict = ""

    if "2020" in newFileName:
        target_dict = "2020"
    else:
        target_dict = "main"

    if "Rochester" in newFileName:
        data_dict[target_dict]["Rochester"] = newFileName
    elif "Buffalo" in newFileName:
        data_dict[target_dict]["Buffalo"] = newFileName
    elif "Syracuse" in newFileName:
        data_dict[target_dict]["Syracuse"] = newFileName

    print("Averaged " + file + " to " + newFileName)

# After averaging for region, time to merge it all together
for key in data_dict:
    if key == "main":
        final_data_name = "AllAveragedDataNumeric.csv"
    else:
        final_data_name = "AllAveragedData2020Numeric.csv"

    print("Merging " + data_dict[key]["Rochester"] + ", "
          + data_dict[key]["Syracuse"] + ", "
          + data_dict[key]["Buffalo"])
    roch_data = pd.read_csv(og_data_folder + data_dict[key]["Rochester"])
    syracuse_data = pd.read_csv(og_data_folder + data_dict[key]["Syracuse"])
    buffalo_data = pd.read_csv(og_data_folder + data_dict[key]["Buffalo"])

    buff_syr_data = pd.merge(syracuse_data, buffalo_data, on="day", suffixes=("_syr", "_buf"))
    all_data = pd.merge(roch_data, buff_syr_data, on="day")

    all_data.to_csv(og_data_folder + final_data_name)
    print("Merged Success - " + final_data_name)
