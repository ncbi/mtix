import copy
import json
import os.path
import pandas as pd


WORKING_DIR="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/create_test_set_subheading_predictions"


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def create_result_lookup(path):
    lookup = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            pmid, dui, qui, score = line.split("\t")
            pmid = int(pmid)
            score = float(score)
            if pmid not in lookup:
                lookup[pmid] = {}
            if dui not in lookup[pmid]:
                lookup[pmid][dui] = {}
            if qui:
                lookup[pmid][dui][qui] = score
    return lookup


def main():
    mesh_heading_predictions_path =    os.path.join(WORKING_DIR, "val_set_2017-2023_Listwise_Avg_Results.json")
    subheading_results_path =          os.path.join(WORKING_DIR, "val_set_2017-2023_Chained_Subheading_Results.tsv")
    subheading_names_path =            os.path.join(WORKING_DIR, "subheading_names_2023_mesh.tsv")
    subheading_predictions_path =      os.path.join(WORKING_DIR, "val_set_2017-2023_Chained_Subheading_Predictions.json")
    
    mesh_heading_predictions = json.load(open(mesh_heading_predictions_path))
    subheading_names = create_lookup(subheading_names_path)
    results_lookup = create_result_lookup(subheading_results_path)

    subheading_predictions = copy.deepcopy(mesh_heading_predictions)
    for citation in subheading_predictions:
        pmid = citation["PMID"]
        for indexing_prediction in citation["Indexing"]:
            if indexing_prediction["Type"].lower() ==  "descriptor" or indexing_prediction["Type"].lower() == "checktag":
                subheadings = []
                indexing_prediction["Subheadings"] = subheadings
                dui = indexing_prediction["ID"]
                if (pmid in results_lookup) and (dui in results_lookup[pmid]):
                    for qui, score in sorted(results_lookup[pmid][dui].items(), key=lambda x: x[1], reverse=True):
                        subheadings.append({
                                "ID": qui,
                                "IM": "NO",
                                "Name": subheading_names[qui],
                                "Reason": f"score: {score:.3f}"})

    json.dump(subheading_predictions, open(subheading_predictions_path, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()