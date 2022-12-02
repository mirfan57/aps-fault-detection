import pymongo
import pandas as pd
import json

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

database_name = "aps"

collection_name = "sensor"

datafile_path = "/config/workspace/aps_failure_training_set1.csv"

if __name__ == "__main__":
    df = pd.read_csv(datafile_path)
    print(f"No. of rows, cols: {df.shape}")
    # convert given df records to json format so that we can dump into mongodb

    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    # df.T.to_json() returns a string object containing dictionary
    # json.loads() takes in a string and returns a json object which is a nested dict, so we use .values()
    print(json_record[1])

    # inserting converted json record into mongodb
    client[database_name][collection_name].insert_many(json_record)