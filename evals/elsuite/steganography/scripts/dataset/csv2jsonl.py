import csv
import json


def csv_to_jsonl(csv_path, jsonl_path):
    json_array = []

    # read csv file
    with open(csv_path, encoding="utf-8") as csvf:
        # load csv file data using csv library's dictionary reader
        csv_reader = csv.DictReader(csvf)

        # convert each csv row into python dict
        for row in csv_reader:
            # append this python dict to json array
            json_array.append(row)

    # convert python jsonArray to JSON String and write to file
    with open(jsonl_path, "w", encoding="utf-8") as jsonf:
        for line in json_array:
            json.dump(line, jsonf)
            jsonf.write("\n")


if __name__ == "__main__":
    csv_path = "dataset.csv"
    jsonl_path = "samples.jsonl"
    csv_to_jsonl(csv_path, jsonl_path)
